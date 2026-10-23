# 📚 PDF 知识库问答机器人

基于 **RAG（检索增强生成）** 架构的智能文档问答系统。上传 PDF 或 Word 文档后，即可针对文档内容提问，回答附参考来源，可核对出处。

> **在线体验**：暂未部署在线 Demo。部署到 Streamlit Cloud 后只需在 Secrets 中填一个 `GEMINI_API_KEY` 即可上线，步骤见文末「Streamlit Cloud 部署」。

---

## ✨ 功能（均已在代码中实现）

- 📄 **多文件上传**：支持 PDF（`PyPDFLoader`）/ Word .docx（`Docx2txtLoader`），可一次上传多个文件
- 🧹 **文本清洗**：去除 Word 图片占位符（U+FFFC）与控制字符、合并多余空行，并过滤清洗后过短的空页
- ✂️ **递归切分**：`RecursiveCharacterTextSplitter`，chunk 1000 字符 / overlap 200，优先按段落、句子切分
- 🗄️ **向量存储**：ChromaDB 持久化到本地 `vectorstore/` 目录；已有库时走增量追加，应用重启后自动加载已入库的向量库
- 🎯 **MMR 检索**：使用最大边际相关性（MMR）而非纯相似度检索，`k=6`、`fetch_k=18`、`lambda_mult=0.7`，减少重复片段、兼顾相关性与多样性
- 📝 **Prompt 约束**：生成层有 5 条明确规则（综合所有片段作答、只依据文档内容、不编造、无依据时诚实说明等），回答后附参考来源
- 🔍 **调试模式**：侧边栏可开启，显示每个检索片段的来源文件、页码、字符长度和原文，方便定位“检索错了还是生成错了”
- 🧹 **知识库管理**：可一键清空向量库 / 清空对话
- 🌐 **中英双语界面**：侧边栏一键切换
- 📊 **离线评测脚本**：`eval_rag.py` + `testset.json`（19 条：15 条可答 / 4 条应拒答），统计 Recall@k、MRR@k、检索耗时 P95，并可选 LLM 判定回答忠实度与拒答正确率

## 📏 评测结果（注意口径，勿夸大）

来自仓库内的 `eval_report.md`（2026-10-22 实测，MMR k=6）：

| 指标 | 数值 |
|---|---|
| Recall@6 | 100.0% |
| MRR@6 | 0.678 |
| 检索耗时 P95 | 0.79s（792 ms） |

⚠️ **这组数字是在示例小语料上测的**：当时知识库只有 **1 份文档、4 个片段**，片段总数小于 k 的量级，Recall 必然偏高，**不能代表大语料下的效果**。评测脚本本身就是为扩充语料后重测准备的；如需引用这组指标，请连同上述语料规模一起说明。

复现：先在应用中上传并处理文档，再运行 `python eval_rag.py --run`（细节见脚本头部注释）。

## 🛠️ 技术栈

| 组件 | 技术 |
|---|---|
| 前端界面 | Streamlit |
| RAG 框架 | LangChain（`create_retrieval_chain` + `create_stuff_documents_chain`） |
| 向量数据库 | ChromaDB（本地持久化） |
| 语言模型 | Google Gemini，默认 `gemini-3.6-flash`（`config.py` 中 `GEMINI_MODEL` 可配） |
| Embedding | Google Gemini，默认 `models/gemini-embedding-2`（`EMBEDDING_MODEL` 可配） |

## 📁 项目结构

```text
app.py           # Streamlit 界面（含调试模式、中英双语）
rag_engine.py    # RAG 引擎：加载/清洗/切分/向量化/MMR 检索/问答链
config.py        # 配置：API Key、模型、切分与检索参数（Secrets 优先，其次 .env）
eval_rag.py      # 评测脚本（Recall@k / MRR@k / P95 / 拒答与忠实度判定）
testset.json     # 评测集（19 条）
eval_report.md   # 最近一次评测报告
requirements.txt # Python 依赖
.streamlit/      # Streamlit 配置与 Secrets 示例（真实 secrets.toml 已被 .gitignore 忽略）
```

## 🚀 本地运行

```bash
# 1. 克隆项目
git clone https://github.com/xiashuidaolonggou/pdf-qa-bot.git
cd pdf-qa-bot

# 2. 创建虚拟环境
python -m venv venv
# Windows: venv\Scripts\activate
# macOS / Linux: source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置 API Key（二选一）
# 本地：在项目根目录新建 .env 文件，填入：
# GEMINI_API_KEY=你的 Gemini API Key
# 或：复制 .streamlit/secrets.toml.example 为 .streamlit/secrets.toml 后填入

# 5. 启动应用
streamlit run app.py
```

未配置 `GEMINI_API_KEY` 时应用不会崩溃，会在页面上提示先去配置 Secrets / .env 后停止加载。

## ☁️ Streamlit Cloud 部署

1. 将本仓库推送到你的 GitHub（本项目已是公开仓库，可直接使用，也可 Fork）。
2. 登录 [Streamlit Community Cloud](https://share.streamlit.io)，点 **New App**。
3. 选择仓库 `pdf-qa-bot`、分支 `master`，主文件选 `app.py`；Advanced settings 中 Python 版本选 3.11 或 3.12。
4. 在 Advanced settings → **Secrets** 中只填一项（格式见 `.streamlit/secrets.toml.example`）：

   ```toml
   GEMINI_API_KEY = "你的 Gemini API Key"
   ```

5. 点击 **Deploy**。代码中其余参数（模型名、切分、TOP_K 等）都有默认值，一般无需在 Secrets 中再填。

部署注意事项（都是当前实现的真实边界）：

- Streamlit Cloud 的本地磁盘不是永久存储：应用重启/重新部署后，`vectorstore/` 与 `uploads/` 可能被清空，需要重新上传文档入库。
- 本系统依赖 Google Gemini 在线 API，离线环境无法使用；扫描版 PDF（纯图片）没有 OCR，提取不到文字，需要先自行 OCR 再上传。
- 同一份文档重复上传会重复入库，系统目前没有做文档去重。

## ⚠️ 已知限制

- 评测语料规模很小（见上文评测口径），混合检索、rerank 等优化尚未实现，不要据此宣称大规模检索效果。
- 已处理文件名列表只存在会话状态中，应用重启后列表不恢复（向量库本身会恢复）。
