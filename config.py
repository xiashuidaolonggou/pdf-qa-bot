"""
配置管理模块 - 云端部署版
支持本地 .env 文件 和 Streamlit Cloud Secrets 两种方式读取配置
"""
import os
from dotenv import load_dotenv

# 加载本地 .env 文件（本地开发时使用，云端会被 Streamlit Secrets 覆盖）
load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """
    优先从 Streamlit Secrets 读取（云端部署），
    其次从环境变量/.env 读取（本地开发）
    """
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)


# ===== Gemini API 配置 =====
# 注意：API Key 从 Streamlit Secrets 或 .env 读取，不硬编码在代码里
GEMINI_API_KEY = _get_secret("GEMINI_API_KEY")
GEMINI_MODEL = _get_secret("GEMINI_MODEL", "gemini-3.6-flash")
EMBEDDING_MODEL = _get_secret("EMBEDDING_MODEL", "models/gemini-embedding-2")

# ===== 文本切分配置 =====
CHUNK_SIZE = int(_get_secret("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(_get_secret("CHUNK_OVERLAP", "50"))

# ===== 检索配置 =====
TOP_K = int(_get_secret("TOP_K", "4"))

# ===== 路径配置 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")

# 确保目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
