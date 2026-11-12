# -*- coding: utf-8 -*-
"""
学术PDF摘要与问答助手
=====================
一个基于 Streamlit 的学术论文 PDF 阅读与问答网页应用。

功能：
    1. 上传 PDF 文件（仅支持 .pdf 格式）
    2. 使用 pypdf 提取 PDF 文本，并预览前 1000 个字符
    3. 接入 DeepSeek 大模型，实现基于论文内容的 AI 问答

依赖说明：
    - streamlit : 网页框架
    - pypdf     : PDF 文本提取
    - openai    : 调用 DeepSeek API（DeepSeek 兼容 OpenAI 接口格式）

运行方式（在命令行中执行）：
    1. 安装依赖：pip install streamlit pypdf openai
    2. 启动应用：streamlit run app.py
"""

import streamlit as st
from pypdf import PdfReader
from openai import OpenAI


# DeepSeek API 的固定配置（Key 不在此处配置，由用户在页面中自行填写）
DEEPSEEK_BASE_URL = "https://api.deepseek.com"  # DeepSeek 的接口地址
DEEPSEEK_MODEL = "deepseek-chat"                # 使用的模型名称


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


def call_deepseek(api_key: str, pdf_text: str, question: str) -> str:
    """
    调用 DeepSeek 大模型进行问答。

    参数:
        api_key  : 用户在页面中填入的 DeepSeek API Key
        pdf_text : 截取后的论文文本（作为 System Prompt）
        question : 用户提出的问题（作为 User Prompt）

    返回:
        str: 大模型返回的回答内容
    """
    # 创建 OpenAI 兼容客户端，指向 DeepSeek 的接口地址
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    # 构造 System Prompt：让模型扮演学术助手，并给出论文内容作为背景知识
    system_prompt = (
        "你是一个专业的学术助手，请基于以下论文内容回答问题：\n"
        f"{pdf_text}"
    )

    # 调用对话补全接口
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        stream=False,  # 使用非流式输出，一次性拿到完整结果
    )

    # 取出模型返回的文本内容
    return response.choices[0].message.content


def main():
    # ==================== 页面顶部大标题 ====================
    st.title("学术PDF智能助手")
    # 在标题下方加一行灰色小字说明
    st.caption("上传学术论文 PDF，快速预览全文并提问")

    # ==================== 左侧边栏：文件上传 + API Key 输入 ====================
    # 使用 with st.sidebar 把组件放进侧边栏
    with st.sidebar:
        st.header("📁 上传文件")
        # 文件上传组件：仅允许上传 PDF 格式文件
        uploaded_file = st.file_uploader(
            "请上传 PDF 文件",
            type=["pdf"],                  # 只接受 .pdf 后缀的文件
            help="仅支持 .pdf 格式文件",    # 鼠标悬停时显示的提示
        )

        # API Key 输入框（type="password" 使输入内容以圆点遮罩显示）
        # 重要：Key 由用户自行填写，切勿硬编码到代码中
        st.header("🔑 API 设置")
        api_key = st.text_input(
            "请输入 DeepSeek API Key",
            type="password",
            help="在 https://platform.deepseek.com 获取你的 API Key",
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
        # ---- 第一步：校验前置条件 ----
        if uploaded_file is None:
            # 未上传 PDF
            st.warning("⚠️ 请先在左侧上传 PDF 文件。")
        elif not api_key.strip():
            # 未填写 API Key
            st.warning("⚠️ 请先在左侧填入 DeepSeek API Key。")
        elif not question.strip():
            # 问题为空
            st.warning("⚠️ 请先输入你的问题。")
        else:
            # ---- 第二步：读取已提取的论文文本 ----
            full_text = st.session_state.get("pdf_text", "")
            if not full_text:
                # 理论上不会走到这里，因为前面已判断上传了文件；
                # 但为稳妥起见仍做一次兜底判断
                st.warning("⚠️ 未提取到有效的论文文本，请重新上传 PDF。")
            else:
                # 为防止 Token 溢出，只截取前 5000 个字符作为上下文
                pdf_context = full_text[:5000]

                # 显示等待动画
                with st.spinner("AI 正在深度思考中..."):
                    try:
                        # ---- 第三步：调用 DeepSeek API ----
                        answer = call_deepseek(api_key, pdf_context, question)

                        # ---- 第四步：展示大模型的回复 ----
                        st.markdown("### 💡 AI 回复")
                        st.markdown(answer)

                    except Exception as e:
                        # ---- 异常捕获：友好提示 ----
                        # 常见错误：
                        #   - API Key 填错 / 无效 -> 鉴权失败 (401)
                        #   - 网络不通 / 超时
                        #   - 账户余额不足 (402)
                        st.error("❌ 调用 AI 接口时出错，请检查以下事项：\n"
                                 "1. API Key 是否正确（注意不要有多余空格）；\n"
                                 "2. 网络连接是否正常；\n"
                                 "3. DeepSeek 账户余额是否充足。\n\n"
                                 f"错误详情：{e}")


# 程序入口
if __name__ == "__main__":
    main()
