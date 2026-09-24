# 📚 PDF 知识库问答机器人

基于 **RAG（检索增强生成）** 架构的智能文档问答系统。上传 PDF 或 Word 文档，即可向 AI 提问关于文档内容的问题。

🌐 **在线体验**：[点击访问](https://your-app.streamlit.app)（部署后替换此链接）

---

## ✨ 功能

- 📄 支持 PDF / Word (.docx) 文档上传
- 🔍 语义检索（ChromaDB 向量数据库）
- 🤖 AI 问答（Google Gemini API）
- 📖 回答附带原文参考来源
- 🌐 中英文界面切换

## 🛠️ 技术栈

| 组件 | 技术 |
|------|------|
| 前端界面 | Streamlit |
| RAG 框架 | LangChain |
| 向量数据库 | ChromaDB |
| 语言模型 | Google Gemini 3.6 Flash |
| Embedding | Google Gemini Embedding 2 |

## 🚀 本地运行

```bash
# 1. 克隆项目
git clone https://github.com/your-username/pdf-qa-bot.git
cd pdf-qa-bot

# 2. 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置 API Key
# 新建 .env 文件，填入：
# GEMINI_API_KEY=你的 Key

# 5. 启动应用
streamlit run app.py
```

## ☁️ Streamlit Cloud 部署

1. Fork 本仓库到你的 GitHub
2. 登录 [Streamlit Community Cloud](https://share.streamlit.io)
3. New App → 选择此仓库 → 主文件选 `app.py`
4. Advanced settings → Secrets 中填入：
   ```toml
   GEMINI_API_KEY = "你的 Gemini API Key"
   ```
5. 点击 Deploy！
