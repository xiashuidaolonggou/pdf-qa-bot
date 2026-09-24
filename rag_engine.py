"""
RAG 核心引擎
负责文档加载、文本切分、向量化存储、语义检索、问答生成
"""
import os
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

import config


# ===== 1. 文档加载 =====
import re

def _clean_text(text: str) -> str:
    """
    清理文档提取的文本：
    - 去除图片/对象占位符（\ufffc，Word 中图片的 Unicode 占位符）
    - 合并多余空白行
    """
    # 去掉 Word 图片占位符 U+FFFC
    text = text.replace("\ufffc", "")
    # 去掉 PDF 中常见的无意义乱码字符（连续多个问号、方块等）
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # 合并连续 3 个以上空行为 2 个
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_document(file_path: str) -> List[Document]:
    """
    根据文件扩展名自动选择合适的 Loader 加载文档，并清理文本

    参数:
        file_path: 文件路径
    返回:
        Document 列表，每个 Document 包含 page_content 和 metadata
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}，目前支持 .pdf 和 .docx")

    documents = loader.load()

    # 清理每个 Document 的文本内容
    for doc in documents:
        doc.page_content = _clean_text(doc.page_content)

    # 过滤掉清理后内容为空的页
    documents = [doc for doc in documents if len(doc.page_content.strip()) > 10]

    return documents



# ===== 2. 文本切分 =====
def split_documents(documents: List[Document]) -> List[Document]:
    """
    将文档切分成小块，方便向量化和检索
    
    使用 RecursiveCharacterTextSplitter：
    - 优先按段落、句子、单词的顺序切分
    - chunk_size: 每块最大字符数
    - chunk_overlap: 相邻块的重叠字符数（保证上下文连贯）
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


# ===== 3. Embedding 模型 =====
def get_embeddings():
    """
    获取 Google 的文本 Embedding 模型
    用于将文本转换为向量（一串数字），语义相近的文本向量也相近
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GEMINI_API_KEY,
    )
    return embeddings


# ===== 4. 向量存储 =====
def create_vectorstore(chunks: List[Document]) -> Chroma:
    """
    将文本块转换为向量并存入 ChromaDB
    
    ChromaDB 是一个向量数据库：
    - 把文本转成向量（Embedding）后存储
    - 支持按语义相似度检索
    - persist_directory: 持久化到磁盘，重启不丢失
    """
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.VECTORSTORE_DIR,
    )
    return vectorstore


def load_vectorstore() -> Chroma:
    """加载已有的向量数据库"""
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=config.VECTORSTORE_DIR,
        embedding_function=embeddings,
    )
    return vectorstore


def add_to_vectorstore(chunks: List[Document], vectorstore: Chroma) -> Chroma:
    """向已有的向量数据库中添加新文档"""
    vectorstore.add_documents(chunks)
    return vectorstore


# ===== 5. 问答链 =====
def get_qa_chain(vectorstore: Chroma):
    """
    构建完整的 RAG 问答链：
    1. 用户提问 → 转为向量
    2. 在 ChromaDB 中检索最相关的 K 个文档片段
    3. 将文档片段 + 用户问题组合成 Prompt
    4. 调用 Gemini 生成回答
    """
    # 初始化 Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GEMINI_API_KEY,
    )

    # 构建 Prompt 模板（改进版：明确要求综合所有片段作答）
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的文档问答助手。以下是从文档中检索到的多个相关片段，请综合所有片段的内容来回答用户的问题。

重要规则：
1. 请仔细阅读【所有】参考片段，即使某个片段只包含章节标题，其相邻片段可能包含答案
2. 综合多个片段的信息给出完整回答，不要因为单个片段内容不完整就说"文档中没有相关信息"
3. 只根据提供的文档内容回答，不要编造文档中没有的信息
4. 回答要简洁、准确、有条理，如果合适请用列表或分点组织
5. 如果所有片段都确实不含相关信息，再诚实说明

参考文档片段：
{context}"""),
        ("human", "{input}"),
    ])

    # 创建文档问答链
    document_chain = create_stuff_documents_chain(llm, prompt)

    # 使用 MMR（最大边际相关性）检索，确保检索结果多样性，
    # 避免返回内容高度重复的 chunk，同时覆盖标题和正文
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": config.TOP_K,
            "fetch_k": config.TOP_K * 3,  # 先取更多候选，再从中挑多样的
            "lambda_mult": 0.7,           # 0=最大多样性，1=纯相似度，0.7取平衡
        },
    )

    # 组合成完整的 RAG 链
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain


# ===== 6. 处理完整流程 =====
def process_uploaded_files(file_paths: List[str]) -> Chroma:
    """
    处理上传的文件：加载 → 切分 → 向量化 → 存储
    
    参数:
        file_paths: 文件路径列表
    返回:
        向量数据库实例
    """
    all_chunks = []
    
    for file_path in file_paths:
        # 加载文档
        documents = load_document(file_path)
        # 切分文本
        chunks = split_documents(documents)
        all_chunks.extend(chunks)
    
    if not all_chunks:
        raise ValueError("没有从文件中提取到任何文本内容")
    
    # 检查是否已有向量数据库
    if os.path.exists(config.VECTORSTORE_DIR) and os.listdir(config.VECTORSTORE_DIR):
        vectorstore = load_vectorstore()
        add_to_vectorstore(all_chunks, vectorstore)
    else:
        vectorstore = create_vectorstore(all_chunks)
    
    return vectorstore, len(all_chunks)


def ask_question(question: str, vectorstore: Chroma) -> dict:
    """
    向知识库提问
    
    参数:
        question: 用户的问题
        vectorstore: 向量数据库实例
    返回:
        包含 answer 和 context 的字典
    """
    qa_chain = get_qa_chain(vectorstore)
    response = qa_chain.invoke({"input": question})
    
    raw_answer = response["answer"]
    if hasattr(raw_answer, "content"):
        answer_text = str(raw_answer.content)
    else:
        answer_text = str(raw_answer)
    
    return {
        "answer": answer_text,
        "source_documents": response["context"],
    }


def clear_vectorstore():
    """清空向量数据库"""
    import shutil
    import gc
    gc.collect()
    if os.path.exists(config.VECTORSTORE_DIR):
        try:
            shutil.rmtree(config.VECTORSTORE_DIR)
        except Exception:
            for item in os.listdir(config.VECTORSTORE_DIR):
                item_path = os.path.join(config.VECTORSTORE_DIR, item)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                except Exception:
                    pass
        os.makedirs(config.VECTORSTORE_DIR, exist_ok=True)
