# -*- coding: utf-8 -*-
"""
学术PDF摘要与问答助手（V6 正式版）
===================================
一个基于 Streamlit 的学术论文 PDF 阅读与问答网页应用。

功能：
    1. 支持同时上传多个 PDF 文件（accept_multiple_files=True），
       解析后合并全文，再做切块与 TF-IDF 索引
    2. 使用 RAG（检索增强生成）+ 跨语言检索：
       - 中文提问 + 英文论文时，先把中文问题翻译成学术英语
       - 同时用「翻译后的英文问题」和「原中文问题」做双路检索，合并去重取 top-k
       - 把检索到的英文文本块 + 原始中文问题一起交给 DeepSeek 生成回答
    3. ChatGPT 式聊天流界面（st.chat_message + st.chat_input），
       对话历史保存在 st.session_state["messages"]
    4. 侧边栏提供「重置会话与文献」按钮，一键清空所有缓存

RAG 流程简述：
    - RecursiveCharacterTextSplitter 把全文切成每块 1500 字符、重叠 300 字符的文本块
    - 「纯 Python + numpy」实现 TF-IDF 向量化 + 余弦相似度，作为轻量检索器
    - 提问时：中文问题 -> DeepSeek 翻译成英文 -> 双路检索 -> 合并去重 -> 生成回答

依赖说明（命令行安装）：
    pip install streamlit pypdf openai langchain-text-splitters numpy

    - streamlit                 : 网页框架
    - pypdf                     : PDF 文本提取
    - openai                    : 调用 DeepSeek API（兼容 OpenAI 接口格式）
    - langchain-text-splitters  : 文本切分（PyPI 包名带连字符，导入时用下划线）
    - numpy                     : TF-IDF 余弦相似度计算

运行方式：
    streamlit run app.py
"""

import re  # 分词时的正则匹配

import numpy as np  # TF-IDF 向量与余弦相似度计算
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==================== 全局配置 ====================
DEEPSEEK_BASE_URL = "https://api.deepseek.com"  # DeepSeek 接口地址
DEEPSEEK_MODEL = "deepseek-chat"                # 使用的模型名称

CHUNK_SIZE = 1500        # 每个文本块的目标字符数（较大以保留表格/上下文完整性）
CHUNK_OVERLAP = 300      # 相邻文本块之间的重叠字符数
TOP_K = 5                # 检索时返回的最相关文本块数量
MAX_FILE_SIZE_MB = 50    # 单个 PDF 文件的大小上限（MB），超过则跳过并提示


# ==================== 1. PDF 文本提取 ====================
def extract_pdf_text(pdf_file) -> str:
    """
    使用 pypdf 提取单个 PDF 文件的全部文本内容。
    """
    reader = PdfReader(pdf_file)          # 构造 PdfReader 对象
    full_text = []                        # 用列表暂存每一页文本
    for page in reader.pages:             # 逐页遍历
        page_text = page.extract_text()   # 提取当前页文本
        if page_text:                     # 跳过无文本的页面（如纯图片页）
            full_text.append(page_text)
    return "\n\n".join(full_text)         # 各页之间用空行分隔


def extract_all_text(uploaded_files):
    """
    遍历所有上传的 PDF 文件，提取并拼接文本。

    说明：
        - 对每个文件做大小检查与异常捕获，解析失败不影响其他文件
        - 各文件文本之间用文件标记分隔，便于追溯来源

    参数:
        uploaded_files: st.file_uploader 返回的文件对象列表

    返回:
        (combined_text, errors)
        - combined_text: 拼接后的全部文本（可能为空字符串）
        - errors: 解析失败/被跳过的提示信息列表
    """
    all_texts = []
    errors = []

    for f in uploaded_files:
        # ---- 大小检查：文件过大则跳过，避免卡死/内存溢出 ----
        if f.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            errors.append(f"{f.name}：文件过大（{f.size / 1024 / 1024:.1f} MB，"
                          f"超过 {MAX_FILE_SIZE_MB} MB 上限），已跳过。")
            continue

        try:
            text = extract_pdf_text(f)
            if text.strip():
                # 用文件标记区分不同来源
                all_texts.append(f"【文件：{f.name}】\n{text}")
            else:
                errors.append(f"{f.name}：未提取到文本（可能是扫描件或纯图片 PDF）。")
        except Exception as e:
            errors.append(f"{f.name}：解析失败（{e}）。")

    combined_text = "\n\n".join(all_texts)
    return combined_text, errors


# ==================== 2. 文本切分 ====================
def split_text_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
):
    """
    使用 RecursiveCharacterTextSplitter 把全文切分成若干文本块。

    说明：
        - chunk_size=1500：每块约 1500 个字符，较大的块能让表格和上下文更完整
        - chunk_overlap=300：相邻块之间有 300 字符重叠，避免关键句被切开
        - separators 按优先级尝试切分：段落、换行、中文标点、空格，最后按单个字符
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # 优先在语义边界（段落/句子）处切分，尽量避免把完整句子拦腰截断
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        length_function=len,  # 用字符数作为长度度量
    )
    # split_text 返回字符串列表，每个元素是一个文本块
    return text_splitter.split_text(text)


# ==================== 3. 轻量分词 ====================
def tokenize(text: str):
    """
    轻量分词：不依赖第三方中文分词库（如 jieba），用正则 + 规则实现。

    策略：
        - 英文/数字：按连续字母数字串切分，统一转小写
        - 中文：提取中文字符，生成「单字」+「相邻两字 bigram」

    返回:
        list[str]: 分词结果（token 列表）
    """
    tokens = []

    # 英文单词与数字（连续字母数字串）
    tokens.extend(re.findall(r"[a-zA-Z0-9]+", text.lower()))

    # 提取文本中的所有中文字符
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)

    # 单个中文字符作为 token
    tokens.extend(chinese_chars)

    # 相邻两字组合成 bigram，提高中文短语匹配精度
    for i in range(len(chinese_chars) - 1):
        tokens.append(chinese_chars[i] + chinese_chars[i + 1])

    return tokens


# ==================== 4. TF-IDF 检索器（纯 Python + numpy） ====================
class SimpleTfidfRetriever:
    """
    基于 TF-IDF + 余弦相似度的轻量检索器。

    不使用 embedding 模型，也不使用 faiss，只用 numpy 计算，足够轻量。
    """

    def __init__(self, chunks):
        self.chunks = chunks
        # 预先对每个文本块分词
        self.tokenized_chunks = [tokenize(c) for c in chunks]
        # 构建词汇表、IDF 向量和 TF-IDF 矩阵
        self.vocab_index, self.idf, self.matrix = self._build_tfidf_matrix()

    def _build_tfidf_matrix(self):
        """
        构建 TF-IDF 矩阵。

        返回:
            vocab_index : dict，token -> 列索引
            idf         : numpy 向量，每个 token 的 IDF 权重
            matrix      : numpy 矩阵，形状 (文本块数, 词表大小)，已做 L2 归一化
        """
        # ---- 1. 构建词汇表（去重 + 排序，保证顺序稳定） ----
        vocab_set = set()
        for tokens in self.tokenized_chunks:
            vocab_set.update(tokens)
        vocab = sorted(vocab_set)
        vocab_index = {word: i for i, word in enumerate(vocab)}

        n_docs = len(self.chunks)                       # 文本块数量
        n_vocab = len(vocab)                            # 词表大小

        # ---- 2. 统计每个 token 出现在多少个文本块中（文档频率 DF） ----
        df = np.zeros(n_vocab)
        for tokens in self.tokenized_chunks:
            for word in set(tokens):
                df[vocab_index[word]] += 1

        # ---- 3. 计算 IDF（平滑处理，避免除零） ----
        idf = np.log((n_docs + 1) / (df + 1)) + 1.0

        # ---- 4. 计算 TF 矩阵，并乘以 IDF 得到 TF-IDF 矩阵 ----
        tf = np.zeros((n_docs, n_vocab))
        for i, tokens in enumerate(self.tokenized_chunks):
            for word in tokens:
                tf[i, vocab_index[word]] += 1
        tfidf = tf * idf

        # ---- 5. 对每一行做 L2 归一化，使「余弦相似度」等价于「点积」 ----
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True) + 1e-9
        tfidf = tfidf / norms

        return vocab_index, idf, tfidf

    def _query_vector(self, query: str):
        """
        把查询文本转成 L2 归一化的 TF-IDF 向量。

        返回:
            numpy 向量；若查询中的词都不在词表里，返回 None（表示无有效匹配）
        """
        query_tokens = tokenize(query)

        n_vocab = len(self.vocab_index)
        q_tf = np.zeros(n_vocab)
        for word in query_tokens:
            if word in self.vocab_index:
                q_tf[self.vocab_index[word]] += 1

        q_vec = q_tf * self.idf                         # 应用 IDF 权重

        # L2 归一化（先检查原始模长，避免除零）
        raw_norm = np.linalg.norm(q_vec)
        if raw_norm < 1e-8:
            return None
        return q_vec / raw_norm

    def retrieve_indices(self, query: str, k: int = TOP_K):
        """
        检索与查询最相关的 top-k 个文本块，返回「文本块下标 + 相似度」。

        返回:
            list[tuple[int, float]]: [(文本块下标, 相似度得分), ...]，按相似度降序；
            若查询无有效匹配，返回空列表
        """
        q_vec = self._query_vector(query)
        if q_vec is None:
            return []

        # 计算余弦相似度（矩阵已归一化，点积即余弦值）
        sims = self.matrix @ q_vec

        # 取出相似度最高的前 k 个下标
        top_indices = np.argsort(sims)[::-1][:k]

        return [(int(i), float(sims[i])) for i in top_indices]

    def retrieve(self, query: str, k: int = TOP_K):
        """
        检索与查询最相关的 top-k 个文本块，返回「文本内容 + 相似度」。

        返回:
            list[tuple[str, float]]: [(文本块内容, 相似度得分), ...]，按相似度降序
        """
        return [(self.chunks[i], score) for i, score in self.retrieve_indices(query, k)]


# ==================== 5. 问题翻译（中 -> 英） ====================
def translate_to_english(api_key: str, question: str) -> str:
    """
    调用 DeepSeek，把用户的中文问题翻译成学术英语（只输出英文）。
    """
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {
                "role": "system",
                "content": "你是一个专业的学术翻译助手。请把用户的中文学术问题翻译成英文，"
                           "只输出英文翻译，不要输出任何解释或额外内容。",
            },
            {"role": "user", "content": question},
        ],
        stream=False,
    )

    return response.choices[0].message.content.strip()


# ==================== 6. 调用 DeepSeek 生成最终回答 ====================
def call_deepseek(api_key: str, context_text: str, question: str) -> str:
    """
    调用 DeepSeek 大模型，基于检索到的英文文本块，回答用户最初的中文问题。
    """
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    system_prompt = (
        "你是一个专业的学术助手。下面提供了若干与问题相关的英文论文片段。\n"
        "请基于这些英文片段回答用户的问题；如果能在片段中找到依据，请尽量直接引用英文原句。\n"
        "如果片段中没有足够信息，请如实说明，不要编造。\n"
        "请使用中文回答用户的问题。\n\n"
        "【相关英文论文片段】\n"
        f"{context_text}"
    )

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        stream=False,
    )

    return response.choices[0].message.content


# ==================== 7. RAG 问答编排 ====================
def answer_question(api_key: str, retriever, chunks, question: str) -> dict:
    """
    执行完整的 RAG 问答：翻译 -> 双路检索 -> 生成回答。

    返回:
        dict: assistant 消息。格式 {"role": "assistant", "content": ...}，
              成功时额外附带 "english_question" 与 "top_chunks" 调试信息。
    """
    # ---- 1. 翻译中文问题为英文 ----
    with st.spinner("正在将问题翻译成英文..."):
        try:
            english_question = translate_to_english(api_key, question)
        except Exception as e:
            return {
                "role": "assistant",
                "content": f"❌ 问题翻译失败，请检查 API Key 与网络。\n\n错误详情：{e}",
            }

    # ---- 2. 双路检索 + 3. 生成回答 ----
    with st.spinner("AI 正在深度思考中..."):
        try:
            # 双路检索：英文问题 + 原中文问题，防止翻译偏差导致漏检
            english_results = retriever.retrieve_indices(english_question, k=TOP_K)
            chinese_results = retriever.retrieve_indices(question, k=TOP_K)

            # 合并去重：同一个文本块（按下标）只保留相似度更高的一次
            merged = {}
            for idx, score in english_results + chinese_results:
                if idx not in merged or score > merged[idx]:
                    merged[idx] = score

            # 若两路都没命中（几乎不可能），退化为取前 TOP_K 块兜底
            if not merged:
                top_pairs = [(i, 0.0) for i in range(min(TOP_K, len(chunks)))]
            else:
                top_pairs = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K]

            top_chunks = [(retriever.chunks[idx], score) for idx, score in top_pairs]
            context_text = "\n\n---\n\n".join(chunk for chunk, _ in top_chunks)

            answer = call_deepseek(api_key, context_text, question)
            return {
                "role": "assistant",
                "content": answer,
                "english_question": english_question,
                "top_chunks": top_chunks,
            }

        except Exception as e:
            return {
                "role": "assistant",
                "content": ("❌ 调用 AI 接口时出错，请检查以下事项：\n"
                            "1. API Key 是否正确（注意不要有多余空格）；\n"
                            "2. 网络连接是否正常；\n"
                            "3. DeepSeek 账户余额是否充足。\n\n"
                            f"错误详情：{e}"),
            }


# ==================== 8. 渲染单条消息 ====================
def render_message(msg: dict):
    """
    渲染一条聊天消息。若为带调试信息的 assistant 消息，额外展示检索到的原文片段。
    """
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # 仅对携带调试信息的 assistant 消息展示折叠面板
        if msg["role"] == "assistant" and msg.get("english_question"):
            with st.expander("🔍 查看检索到的原文片段及英文翻译"):
                st.markdown("**🌐 翻译后的英文问题：**")
                st.write(msg["english_question"])
                top_chunks = msg.get("top_chunks", [])
                st.markdown(f"**📚 检索到的原文片段（共 {len(top_chunks)} 块）：**")
                for i, (chunk, score) in enumerate(top_chunks, 1):
                    st.markdown(f"**片段 {i}（相似度 {score:.4f}）**")
                    st.text(chunk)


# ==================== 9. 主函数 ====================
def main():
    # ==================== 页面顶部大标题 ====================
    st.title("学术PDF智能助手")
    st.caption("上传学术论文 PDF，支持中文提问、跨语言双路检索英文全文")

    # ==================== 左侧边栏：文件上传 + API Key + 重置 ====================
    with st.sidebar:
        st.header("📁 上传文件")
        uploaded_files = st.file_uploader(
            "请上传一个或多个 PDF 文件",
            type=["pdf"],
            accept_multiple_files=True,   # 允许一次上传多个文件
            help="支持同时上传多个 PDF，解析后合并全文",
        ) or []                            # 无文件时返回空列表，便于后续统一处理

        st.header("🔑 API 设置")
        api_key = st.text_input(
            "请输入 DeepSeek API Key",
            type="password",
            help="在 https://platform.deepseek.com 获取你的 API Key",
        )

        st.divider()
        # 重置按钮：清空所有缓存并重新开始
        if st.button("🗑️ 重置会话与文献"):
            st.session_state.clear()       # 清空 session_state 中的所有缓存
            st.rerun()                      # 重新运行脚本，回到初始状态

    # ==================== 上半部分：PDF 文本预览 ====================
    st.subheader("📄 PDF文本预览")

    if uploaded_files:
        # 用「文件集合指纹」判断上传的文件是否发生变化，避免每次交互都重复解析
        file_key = tuple(sorted((f.name, f.size) for f in uploaded_files))

        if (
            "current_file_key" not in st.session_state
            or st.session_state["current_file_key"] != file_key
        ):
            with st.spinner("正在解析 PDF 并构建索引，请稍候..."):
                full_text, errors = extract_all_text(uploaded_files)

                # 切分文本块
                chunks = split_text_into_chunks(full_text) if full_text else []

                # 构建检索器（TF-IDF）
                retriever = SimpleTfidfRetriever(chunks) if chunks else None

            # 全部存入 session_state，供聊天区复用
            st.session_state["pdf_text"] = full_text
            st.session_state["chunks"] = chunks
            st.session_state["retriever"] = retriever
            st.session_state["current_file_key"] = file_key
            st.session_state["parse_errors"] = errors

        # 从 session_state 读取解析结果
        full_text = st.session_state.get("pdf_text", "")
        chunks = st.session_state.get("chunks", [])
        errors = st.session_state.get("parse_errors", [])

        # 展示解析错误/警告（文件过大、解析失败等）
        for err in errors:
            st.warning(f"⚠️ {err}")

        if full_text:
            st.text_area("文本预览（前 1000 字符）", value=full_text[:1000], height=300)
            st.info(f"已解析 {len(uploaded_files)} 个文件，全文共 {len(full_text)} 个字符，"
                    f"切分为 {len(chunks)} 个文本块。")
        else:
            st.warning("未能从上传的 PDF 中提取到任何文本，请检查文件是否为扫描件或已损坏。")
    else:
        st.info("请先在左侧上传 PDF 文件，文本预览将显示在这里。")

    # ==================== 下半部分：聊天式 AI 问答 ====================
    st.divider()
    st.subheader("🤖 AI 分析与问答")

    # 初始化聊天记录（首次运行）
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # 渲染历史聊天记录
    for msg in st.session_state["messages"]:
        render_message(msg)

    # 接收新输入（st.chat_input 提交后会自动清空输入框）
    if prompt := st.chat_input("请输入你的问题，例如：这篇论文的核心观点是什么？"):
        # ---- 记录并渲染用户消息 ----
        user_msg = {"role": "user", "content": prompt}
        st.session_state["messages"].append(user_msg)
        render_message(user_msg)

        # ---- 校验前置条件，生成 assistant 回复 ----
        if not uploaded_files:
            assistant_msg = {"role": "assistant", "content": "⚠️ 请先在左侧上传 PDF 文件。"}
        elif not api_key.strip():
            assistant_msg = {"role": "assistant", "content": "⚠️ 请先在左侧填入 DeepSeek API Key。"}
        else:
            retriever = st.session_state.get("retriever")
            chunks = st.session_state.get("chunks", [])
            if retriever is None or not chunks:
                assistant_msg = {"role": "assistant",
                                 "content": "⚠️ 未提取到有效的论文文本，请重新上传 PDF。"}
            else:
                # 执行完整 RAG 问答
                assistant_msg = answer_question(api_key, retriever, chunks, prompt)

        # ---- 记录并渲染 assistant 消息 ----
        st.session_state["messages"].append(assistant_msg)
        render_message(assistant_msg)


if __name__ == "__main__":
    main()
