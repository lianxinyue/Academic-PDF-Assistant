# -*- coding: utf-8 -*-
"""
学术PDF摘要与问答助手
=====================
一个基于 Streamlit 的学术论文 PDF 阅读与问答网页应用。

功能：
    1. 上传 PDF 文件（仅支持 .pdf 格式）
    2. 使用 pypdf 提取 PDF 文本，并预览前 1000 个字符
    3. 提供 AI 问答交互界面（当前为占位实现，后续接入大模型）

运行方式（在命令行中执行）：
    1. 安装依赖：pip install streamlit pypdf
    2. 启动应用：streamlit run app.py
"""

import streamlit as st
from pypdf import PdfReader


def extract_pdf_text(pdf_file) -> str:
    """
    使用 pypdf 提取 PDF 文件的全部文本内容。

    参数:
        pdf_file: streamlit 上传的文件对象（类文件对象，支持 read 方法）

    返回:
        str: 提取出的完整文本内容（各页之间用换行符分隔）
    """
    # 构造 PdfReader 对象，直接传入二进制文件对象即可
    reader = PdfReader(pdf_file)

    # 用列表暂存每一页的文本，最后再统一拼接，效率更高
    full_text = []

    # 逐页遍历 PDF 页面
    for page in reader.pages:
        # 提取当前页的文本内容
        page_text = page.extract_text()
        # 有些页面可能没有可提取的文本（例如纯扫描图片页），需要跳过
        if page_text:
            full_text.append(page_text)

    # 用两个换行符连接各页文本，使页面之间有空行分隔
    return "\n\n".join(full_text)


def main():
    # ==================== 页面顶部大标题 ====================
    st.title("学术PDF智能助手")
    # 在标题下方加一行灰色小字说明
    st.caption("上传学术论文 PDF，快速预览全文并提问")

    # ==================== 左侧边栏：文件上传 ====================
    # 使用 with st.sidebar 把组件放进侧边栏
    with st.sidebar:
        st.header("📁 上传文件")
        # 文件上传组件：仅允许上传 PDF 格式文件
        uploaded_file = st.file_uploader(
            "请上传 PDF 文件",
            type=["pdf"],                  # 只接受 .pdf 后缀的文件
            help="仅支持 .pdf 格式文件",    # 鼠标悬停时显示的提示
        )

    # ==================== 主体区域上半部分：PDF 文本预览 ====================
    st.subheader("📄 PDF文本预览")

    if uploaded_file is not None:
        # 用一个「文件指纹」判断用户是否上传了新文件，
        # 避免每次页面交互（如点击按钮）都重新解析 PDF，提升性能
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"

        # 若当前文件与上次处理的不同，才重新提取文本
        if (
            "current_file_key" not in st.session_state
            or st.session_state["current_file_key"] != file_key
        ):
            # 显示加载动画，提示用户正在处理
            with st.spinner("正在提取 PDF 文本，请稍候..."):
                full_text = extract_pdf_text(uploaded_file)

            # 将提取结果保存到 session_state，供下方问答区域复用
            st.session_state["pdf_text"] = full_text
            st.session_state["current_file_key"] = file_key

        # 从 session_state 中读取已提取的全文
        full_text = st.session_state.get("pdf_text", "")

        if full_text:
            # 截取前 1000 个字符用于预览
            preview_text = full_text[:1000]

            # 在可滚动文本框中展示预览内容
            # height 设置文本框高度，内容超出后会自动出现滚动条
            st.text_area(
                "文本预览（前 1000 字符）",
                value=preview_text,
                height=300,
            )

            # 显示全文总字符数，方便用户了解文档规模
            st.info(f"已成功提取全文，共 {len(full_text)} 个字符。")
        else:
            # 提取结果为空，通常是扫描件（纯图片）PDF
            st.warning("未能从该 PDF 中提取到文本，可能是扫描件（纯图片）PDF，需使用 OCR 技术。")
    else:
        # 尚未上传文件时的占位提示
        st.info("请先在左侧上传 PDF 文件，文本预览将显示在这里。")

    # ==================== 主体区域下半部分：AI 分析与问答 ====================
    st.divider()  # 添加一条分隔线，区隔上下两部分
    st.subheader("🤖 AI分析与问答")

    # 文本输入框，供用户输入问题
    question = st.text_input(
        "请输入你的问题",
        placeholder="例如：这篇论文的核心观点是什么？",
    )

    # “发送”按钮，使用 primary 类型让其更醒目
    if st.button("发送", type="primary"):
        # 校验：问题不能为空
        if not question.strip():
            st.warning("请先输入问题再发送。")
        else:
            # ============================================================
            # 占位实现：暂未接入真实大模型 API
            # 后续接入时，可在此处调用大模型接口，
            # 将 question（问题）与 st.session_state["pdf_text"]（全文）一起传入
            # ============================================================
            st.write("正在思考中...")


# 程序入口
if __name__ == "__main__":
    main()
