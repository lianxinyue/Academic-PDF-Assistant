# -*- coding: utf-8 -*-
"""
学术PDF摘要与问答助手（RAG + 跨语言检索版）
==============================================
功能：
    1. 上传 PDF 文件（仅支持 .pdf 格式）
    2. 使用 pypdf 提取 PDF 文本，并预览前 1000 个字符
    3. RAG（检索增强生成）+ 跨语言检索：
       - 中文提问 + 英文论文时，先把中文问题翻译成学术英语
       - 用英文问题做 TF-IDF 检索，命中对应英文文本块
       - 把英文文本块 + 原始中文问题一起交给 DeepSeek 生成回答
"""

import re
import numpy as np
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3


def extract_pdf_text(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    full_text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text.append(page_text)
    return "\n\n".join(full_text)


def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        length_function=len,
    )
    return text_splitter.split_text(text)


def tokenize(text: str):
    """轻量分词：英文数字串 + 中文单字/二字 bigram。"""
    tokens = []
    tokens.extend(re.findall(r"[a-zA-Z0-9]+", text.lower()))
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    tokens.extend(chinese_chars)
    for i in range(len(chinese_chars) - 1):
        tokens.append(chinese_chars[i] + chinese_chars[i + 1])
    return tokens


class SimpleTfidfRetriever:
    """基于 TF-IDF + 余弦相似度的轻量检索器（纯 Python + numpy）。"""

    def __init__(self, chunks):
        self.chunks = chunks
        self.tokenized_chunks = [tokenize(c) for c in chunks]
        self.vocab_index, self.idf, self.matrix = self._build_tfidf_matrix()

    def _build_tfidf_matrix(self):
        vocab_set = set()
        for tokens in self.tokenized_chunks:
            vocab_set.update(tokens)
        vocab = sorted(vocab_set)
        vocab_index = {w: i for i, w in enumerate(vocab)}

        n_docs, n_vocab = len(self.chunks), len(vocab)

        df = np.zeros(n_vocab)
        for tokens in self.tokenized_chunks:
            for w in set(tokens):
                df[vocab_index[w]] += 1

        idf = np.log((n_docs + 1) / (df + 1)) + 1.0

        tf = np.zeros((n_docs, n_vocab))
        for i, tokens in enumerate(self.tokenized_chunks):
            for w in tokens:
                tf[i, vocab_index[w]] += 1
        tfidf = tf * idf

        norms = np.linalg.norm(tfidf, axis=1, keepdims=True) + 1e-9
        tfidf = tfidf / norms

        return vocab_index, idf, tfidf

    def retrieve(self, query: str, k: int = TOP_K):
        query_tokens = tokenize(query)
        n_vocab = len(self.vocab_index)
        q_tf = np.zeros(n_vocab)
        for w in query_tokens:
            if w in self.vocab_index:
                q_tf[self.vocab_index[w]] += 1

        q_vec = q_tf * self.idf
        q_norm = np.linalg.norm(q_vec) + 1e-9
        q_vec = q_vec / q_norm

        if q_norm < 1e-8:
            return [(self.chunks[i], 0.0) for i in range(min(k, len(self.chunks)))]

        sims = self.matrix @ q_vec
        top_indices = np.argsort(sims)[::-1][:k]
        return [(self.chunks[i], float(sims[i])) for i in top_indices]


def translate_to_english(api_key: str, question: str) -> str:
    """调用 DeepSeek，把中文问题翻译成学术英语（只输出英文）。"""
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


def call_deepseek(api_key: str, context_text: str, question: str) -> str:
    """基于检索到的英文文本块，回答用户最初的中文问题。"""
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


def main():
    st.title("学术PDF智能助手")
    st.caption("上传学术论文 PDF，支持中文提问、跨语言检索英文全文")

    with st.sidebar:
        st.header("📁 上传文件")
        uploaded_file = st.file_uploader("请上传 PDF 文件", type=["pdf"], help="仅支持 .pdf 格式文件")

        st.header("🔑 API 设置")
        api_key = st.text_input("请输入 DeepSeek API Key", type="password",
                                help="在 https://platform.deepseek.com 获取你的 API Key")

    st.subheader("📄 PDF文本预览")

    if uploaded_file is not None:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if ("current_file_key" not in st.session_state
                or st.session_state["current_file_key"] != file_key):
            with st.spinner("正在提取 PDF 文本并构建索引，请稍候..."):
                full_text = extract_pdf_text(uploaded_file)
                chunks = split_text_into_chunks(full_text) if full_text else []
                retriever = SimpleTfidfRetriever(chunks) if chunks else None
            st.session_state["pdf_text"] = full_text
            st.session_state["chunks"] = chunks
            st.session_state["retriever"] = retriever
            st.session_state["current_file_key"] = file_key

        full_text = st.session_state.get("pdf_text", "")
        chunks = st.session_state.get("chunks", [])

        if full_text:
            st.text_area("文本预览（前 1000 字符）", value=full_text[:1000], height=300)
            st.info(f"已提取全文共 {len(full_text)} 个字符，切分为 {len(chunks)} 个文本块。")
        else:
            st.warning("未能从该 PDF 中提取到文本，可能是扫描件（纯图片）PDF，需使用 OCR 技术。")
    else:
        st.info("请先在左侧上传 PDF 文件，文本预览将显示在这里。")

    st.divider()
    st.subheader("🤖 AI分析与问答")

    question = st.text_input("请输入你的问题", placeholder="例如：这篇论文的核心观点是什么？")

    if st.button("发送", type="primary"):
        if uploaded_file is None:
            st.warning("⚠️ 请先在左侧上传 PDF 文件。")
        elif not api_key.strip():
            st.warning("⚠️ 请先在左侧填入 DeepSeek API Key。")
        elif not question.strip():
            st.warning("⚠️ 请先输入你的问题。")
        else:
            retriever = st.session_state.get("retriever")
            chunks = st.session_state.get("chunks", [])
            if retriever is None or not chunks:
                st.warning("⚠️ 未提取到有效的论文文本，请重新上传 PDF。")
            else:
                # 1) 中文问题 -> 英文
                with st.spinner("正在将问题翻译成英文..."):
                    try:
                        english_question = translate_to_english(api_key, question)
                    except Exception as e:
                        st.error(f"❌ 问题翻译失败，请检查 API Key 与网络。\n\n错误详情：{e}")
                        st.stop()

                # 2) 英文问题 -> 检索英文块
                top_chunks = retriever.retrieve(english_question, k=TOP_K)
                context_text = "\n\n---\n\n".join(chunk for chunk, _ in top_chunks)

                # 3) 英文块 + 原中文问题 -> 生成回答
                with st.spinner("AI 正在深度思考中..."):
                    try:
                        answer = call_deepseek(api_key, context_text, question)

                        with st.expander("🔍 查看检索到的原文片段及英文翻译"):
                            st.markdown("**🌐 翻译后的英文问题：**")
                            st.write(english_question)
                            st.markdown(f"**📚 检索到的原文片段（共 {len(top_chunks)} 块，"
                                        f"来自全文 {len(chunks)} 块）：**")
                            for i, (chunk, score) in enumerate(top_chunks, 1):
                                st.markdown(f"**片段 {i}（相似度 {score:.4f}）**&#8203;")
                                st.text(chunk)

                        st.markdown("### 💡 AI 回复")
                        st.markdown(answer)

                    except Exception as e:
                        st.error("❌ 调用 AI 接口时出错，请检查以下事项：\n"
                                 "1. API Key 是否正确（注意不要有多余空格）；\n"
                                 "2. 网络连接是否正常；\n"
                                 "3. DeepSeek 账户余额是否充足。\n\n"
                                 f"错误详情：{e}")


if __name__ == "__main__":
    main()
