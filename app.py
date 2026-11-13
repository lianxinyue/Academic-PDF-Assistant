# -*- coding: utf-8 -*-
"""
学术PDF摘要与问答助手（V6 正式版）
===================================
功能：
    1. 支持同时上传多个 PDF 文件（accept_multiple_files=True）
    2. RAG（检索增强生成）+ 跨语言检索（中译英 + 双路 TF-IDF 检索）
    3. ChatGPT 式聊天流界面（st.chat_message + st.chat_input）
    4. 侧边栏「重置会话与文献」按钮，一键清空所有缓存
"""

import re
import numpy as np
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
TOP_K = 5
MAX_FILE_SIZE_MB = 50


def extract_pdf_text(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    full_text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text.append(page_text)
    return "\n\n".join(full_text)


def extract_all_text(uploaded_files):
    """遍历所有上传 PDF，提取并拼接文本，返回 (combined_text, errors)。"""
    all_texts = []
    errors = []
    for f in uploaded_files:
        if f.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            errors.append(f"{f.name}：文件过大（{f.size / 1024 / 1024:.1f} MB，"
                          f"超过 {MAX_FILE_SIZE_MB} MB 上限），已跳过。")
            continue
        try:
            text = extract_pdf_text(f)
            if text.strip():
                all_texts.append(f"【文件：{f.name}】\n{text}")
            else:
                errors.append(f"{f.name}：未提取到文本（可能是扫描件或纯图片 PDF）。")
        except Exception as e:
            errors.append(f"{f.name}：解析失败（{e}）。")
    return "\n\n".join(all_texts), errors


def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        length_function=len,
    )
    return text_splitter.split_text(text)


def tokenize(text: str):
    tokens = []
    tokens.extend(re.findall(r"[a-zA-Z0-9]+", text.lower()))
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    tokens.extend(chinese_chars)
    for i in range(len(chinese_chars) - 1):
        tokens.append(chinese_chars[i] + chinese_chars[i + 1])
    return tokens


class SimpleTfidfRetriever:
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
        return vocab_index, idf, tfidf / norms

    def _query_vector(self, query: str):
        query_tokens = tokenize(query)
        n_vocab = len(self.vocab_index)
        q_tf = np.zeros(n_vocab)
        for w in query_tokens:
            if w in self.vocab_index:
                q_tf[self.vocab_index[w]] += 1
        q_vec = q_tf * self.idf
        raw_norm = np.linalg.norm(q_vec)
        if raw_norm < 1e-8:
            return None
        return q_vec / raw_norm

    def retrieve_indices(self, query: str, k: int = TOP_K):
        q_vec = self._query_vector(query)
        if q_vec is None:
            return []
        sims = self.matrix @ q_vec
        top_indices = np.argsort(sims)[::-1][:k]
        return [(int(i), float(sims[i])) for i in top_indices]

    def retrieve(self, query: str, k: int = TOP_K):
        return [(self.chunks[i], score) for i, score in self.retrieve_indices(query, k)]


def translate_to_english(api_key: str, question: str) -> str:
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system",
             "content": "你是一个专业的学术翻译助手。请把用户的中文学术问题翻译成英文，"
                        "只输出英文翻译，不要输出任何解释或额外内容。"},
            {"role": "user", "content": question},
        ],
        stream=False,
    )
    return response.choices[0].message.content.strip()


def call_deepseek(api_key: str, context_text: str, question: str) -> str:
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


def answer_question(api_key: str, retriever, chunks, question: str) -> dict:
    """翻译 -> 双路检索 -> 生成回答，返回 assistant 消息 dict。"""
    with st.spinner("正在将问题翻译成英文..."):
        try:
            english_question = translate_to_english(api_key, question)
        except Exception as e:
            return {"role": "assistant",
                    "content": f"❌ 问题翻译失败，请检查 API Key 与网络。\n\n错误详情：{e}"}

    with st.spinner("AI 正在深度思考中..."):
        try:
            english_results = retriever.retrieve_indices(english_question, k=TOP_K)
            chinese_results = retriever.retrieve_indices(question, k=TOP_K)
            merged = {}
            for idx, score in english_results + chinese_results:
                if idx not in merged or score > merged[idx]:
                    merged[idx] = score
            if not merged:
                top_pairs = [(i, 0.0) for i in range(min(TOP_K, len(chunks)))]
            else:
                top_pairs = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K]
            top_chunks = [(retriever.chunks[idx], score) for idx, score in top_pairs]
            context_text = "\n\n---\n\n".join(chunk for chunk, _ in top_chunks)

            answer = call_deepseek(api_key, context_text, question)
            return {"role": "assistant", "content": answer,
                    "english_question": english_question, "top_chunks": top_chunks}
        except Exception as e:
            return {"role": "assistant",
                    "content": ("❌ 调用 AI 接口时出错，请检查以下事项：\n"
                                "1. API Key 是否正确（注意不要有多余空格）；\n"
                                "2. 网络连接是否正常；\n"
                                "3. DeepSeek 账户余额是否充足。\n\n"
                                f"错误详情：{e}")}


def render_message(msg: dict):
    """渲染一条聊天消息，assistant 消息带调试信息时展示检索片段。"""
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("english_question"):
            with st.expander("🔍 查看检索到的原文片段及英文翻译"):
                st.markdown("**🌐 翻译后的英文问题：**")
                st.write(msg["english_question"])
                top_chunks = msg.get("top_chunks", [])
                st.markdown(f"**📚 检索到的原文片段（共 {len(top_chunks)} 块）：**")
                for i, (chunk, score) in enumerate(top_chunks, 1):
                    st.markdown(f"**片段 {i}（相似度 {score:.4f}）**&#8203;")
                    st.text(chunk)


def main():
    st.title("学术PDF智能助手")
    st.caption("上传学术论文 PDF，支持中文提问、跨语言双路检索英文全文")

    with st.sidebar:
        st.header("📁 上传文件")
        uploaded_files = st.file_uploader(
            "请上传一个或多个 PDF 文件",
            type=["pdf"],
            accept_multiple_files=True,
            help="支持同时上传多个 PDF，解析后合并全文",
        ) or []

        st.header("🔑 API 设置")
        api_key = st.text_input("请输入 DeepSeek API Key", type="password",
                                help="在 https://platform.deepseek.com 获取你的 API Key")

        st.divider()
        if st.button("🗑️ 重置会话与文献"):
            st.session_state.clear()
            st.rerun()

    st.subheader("📄 PDF文本预览")

    if uploaded_files:
        file_key = tuple(sorted((f.name, f.size) for f in uploaded_files))
        if ("current_file_key" not in st.session_state
                or st.session_state["current_file_key"] != file_key):
            with st.spinner("正在解析 PDF 并构建索引，请稍候..."):
                full_text, errors = extract_all_text(uploaded_files)
                chunks = split_text_into_chunks(full_text) if full_text else []
                retriever = SimpleTfidfRetriever(chunks) if chunks else None
            st.session_state["pdf_text"] = full_text
            st.session_state["chunks"] = chunks
            st.session_state["retriever"] = retriever
            st.session_state["current_file_key"] = file_key
            st.session_state["parse_errors"] = errors

        full_text = st.session_state.get("pdf_text", "")
        chunks = st.session_state.get("chunks", [])
        errors = st.session_state.get("parse_errors", [])

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

    st.divider()
    st.subheader("🤖 AI 分析与问答")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        render_message(msg)

    if prompt := st.chat_input("请输入你的问题，例如：这篇论文的核心观点是什么？"):
        user_msg = {"role": "user", "content": prompt}
        st.session_state["messages"].append(user_msg)
        render_message(user_msg)

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
                assistant_msg = answer_question(api_key, retriever, chunks, prompt)

        st.session_state["messages"].append(assistant_msg)
        render_message(assistant_msg)


if __name__ == "__main__":
    main()
