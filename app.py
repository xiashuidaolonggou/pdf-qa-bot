"""
PDF 知识库问答机器人 - Streamlit Web 界面（云端部署版）
基于 RAG 架构，支持 PDF/Word 文档上传与智能问答
支持中英文界面切换
"""
import os
import streamlit as st
import rag_engine
import config


# ===== 语言包 =====
LANG = {
    "zh": {
        "page_title": "📚 PDF 知识库问答",
        "title": "📚 PDF 知识库问答机器人",
        "caption": "上传 PDF 或 Word 文档，然后向 AI 提问关于文档内容的问题",
        "sidebar_header": "📁 文档管理",
        "upload_label": "上传文档（支持 PDF、Word）",
        "upload_help": "可以同时上传多个文件",
        "process_btn": "🚀 处理文档",
        "processing": "⏳ 正在处理文档，请稍候...",
        "process_success": "✅ 处理完成！共生成 {n} 个文本块",
        "process_fail": "❌ 处理失败：{e}",
        "loaded_docs": "📋 已加载的文档",
        "clear_btn": "🗑️ 清空知识库",
        "chat_placeholder": "💬 输入你的问题...",
        "no_kb_warning": "⚠️ 请先在左侧上传并处理文档！",
        "thinking": "🤔 思考中...",
        "sources_label": "📖 参考来源",
        "source_chunk": "片段 {i}：",
        "answer_fail": "❌ 回答生成失败：{e}",
        "lang_label": "🌐 语言 / Language",
        "clear_chat_btn": "🧹 清空对话",
        "status_ready": "✅ 知识库已就绪",
        "status_empty": "📭 知识库为空，请上传文档",
        "db_section": "📊 知识库状态",
        "no_api_key": "⚠️ 未配置 GEMINI_API_KEY，请在 Streamlit Secrets 中设置。",
    },
    "en": {
        "page_title": "📚 PDF Knowledge Base Q&A",
        "title": "📚 PDF Knowledge Base Chatbot",
        "caption": "Upload PDF or Word documents, then ask the AI questions about the content",
        "sidebar_header": "📁 Document Manager",
        "upload_label": "Upload Documents (PDF / Word)",
        "upload_help": "You can upload multiple files at once",
        "process_btn": "🚀 Process Documents",
        "processing": "⏳ Processing documents, please wait...",
        "process_success": "✅ Done! Generated {n} text chunks",
        "process_fail": "❌ Processing failed: {e}",
        "loaded_docs": "📋 Loaded Documents",
        "clear_btn": "🗑️ Clear Knowledge Base",
        "chat_placeholder": "💬 Ask a question...",
        "no_kb_warning": "⚠️ Please upload and process documents first!",
        "thinking": "🤔 Thinking...",
        "sources_label": "📖 Reference Sources",
        "source_chunk": "Chunk {i}:",
        "answer_fail": "❌ Failed to generate answer: {e}",
        "lang_label": "🌐 Language / 语言",
        "clear_chat_btn": "🧹 Clear Chat",
        "status_ready": "✅ Knowledge base ready",
        "status_empty": "📭 Knowledge base empty, please upload documents",
        "db_section": "📊 Knowledge Base Status",
        "no_api_key": "⚠️ GEMINI_API_KEY not configured. Please set it in Streamlit Secrets.",
    },
}

# ===== 页面配置 =====
st.set_page_config(
    page_title="📚 PDF Q&A",
    page_icon="📚",
    layout="wide",
)

# ===== 初始化 Session State =====
if "lang" not in st.session_state:
    st.session_state.lang = "zh"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

L = LANG[st.session_state.lang]

# ===== API Key 检查 =====
if not config.GEMINI_API_KEY:
    st.error(L["no_api_key"])
    st.stop()

# ===== 页面标题 =====
st.title(L["title"])
st.caption(L["caption"])

# ===== 侧边栏 =====
with st.sidebar:
    # 语言切换
    lang_choice = st.radio(
        L["lang_label"],
        options=["中文", "English"],
        index=0 if st.session_state.lang == "zh" else 1,
        horizontal=True,
    )
    new_lang = "zh" if lang_choice == "中文" else "en"
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

    st.divider()

    # 调试模式开关
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    st.session_state.debug_mode = st.checkbox(
        "🔍 调试模式（显示检索到的原始片段）" if st.session_state.lang == "zh" else "🔍 Debug mode (show retrieved chunks)",
        value=st.session_state.debug_mode,
    )

    st.divider()

    st.header(L["sidebar_header"])

    uploaded_files = st.file_uploader(
        L["upload_label"],
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help=L["upload_help"],
    )

    if uploaded_files:
        if st.button(L["process_btn"], type="primary", use_container_width=True):
            saved_paths = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(config.UPLOAD_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_paths.append(file_path)

            with st.spinner(L["processing"]):
                try:
                    vectorstore, chunk_count = rag_engine.process_uploaded_files(saved_paths)
                    st.session_state.vectorstore = vectorstore
                    for uf in uploaded_files:
                        if uf.name not in st.session_state.processed_files:
                            st.session_state.processed_files.append(uf.name)
                    st.success(L["process_success"].format(n=chunk_count))
                    st.rerun()
                except Exception as e:
                    st.error(L["process_fail"].format(e=str(e)))

    st.divider()

    # 知识库状态
    st.subheader(L["db_section"])
    if st.session_state.vectorstore is not None:
        st.success(L["status_ready"])
    else:
        st.info(L["status_empty"])

    # 已加载文档列表
    if st.session_state.processed_files:
        st.markdown(f"**{L['loaded_docs']}**")
        for fname in st.session_state.processed_files:
            st.markdown(f"- 📄 `{fname}`")

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button(L["clear_btn"], use_container_width=True):
                rag_engine.clear_vectorstore()
                st.session_state.vectorstore = None
                st.session_state.processed_files = []
                st.session_state.chat_history = []
                st.rerun()
        with col2:
            if st.button(L["clear_chat_btn"], use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()


# ===== 主区域：聊天界面 =====
L = LANG[st.session_state.lang]

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander(L["sources_label"]):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**{L['source_chunk'].format(i=i)}**")
                    st.markdown(f"> {source}")
                    st.divider()

if prompt := st.chat_input(L["chat_placeholder"]):
    if st.session_state.vectorstore is None:
        st.warning(L["no_kb_warning"])
    else:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner(L["thinking"]):
                try:
                    result = rag_engine.ask_question(prompt, st.session_state.vectorstore)
                    answer = result["answer"]
                    source_docs = result["source_documents"]
                    sources = [doc.page_content for doc in source_docs]

                    st.markdown(answer)

                    # 调试模式：显示原始检索片段
                    if st.session_state.get("debug_mode"):
                        with st.expander("🔍 调试：检索到的原始片段" if st.session_state.lang == "zh" else "🔍 Debug: Retrieved chunks", expanded=True):
                            st.caption(f"共检索到 {len(source_docs)} 个片段（TOP_K={rag_engine.config.TOP_K}，CHUNK_SIZE={rag_engine.config.CHUNK_SIZE}）")
                            for i, doc in enumerate(source_docs, 1):
                                src = doc.metadata.get("source", "未知来源")
                                page = doc.metadata.get("page", "?")
                                st.markdown(f"**片段 {i}** | 来源: `{src}` | 页码: {page} | 长度: {len(doc.page_content)} 字符")
                                st.code(doc.page_content, language=None)
                    elif sources:
                        with st.expander(L["sources_label"]):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**{L['source_chunk'].format(i=i)}**")
                                st.markdown(f"> {source}")
                                st.divider()

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                except Exception as e:
                    st.error(L["answer_fail"].format(e=str(e)))

