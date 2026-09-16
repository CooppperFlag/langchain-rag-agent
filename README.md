# langchain-rag-agent

手写 LangChain 实现的 RAG（检索增强生成）Agent，脱离 Dify 等低代码平台，从零搭建完整链路。

## 项目简介

本项目使用 LangChain + Chroma + BGE-M3 + DeepSeek，实现了一个可对本地文档进行问答的 RAG 系统。
支持 PDF / Markdown / TXT 等多种文档格式。

## 技术栈

- **LangChain**：RAG 编排框架
- **Chroma**：本地向量数据库
- **BAAI/bge-m3**（硅基流动）：Embedding 模型，1024 维
- **DeepSeek-chat**：生成模型
- **Python 3.12** + **venv**

## RAG 流程图

```
文档（PDF/MD/TXT）
     │
     ▼
[PyPDFLoader / TextLoader]  加载
     │
     ▼
[RecursiveCharacterTextSplitter]  切分（chunk_size=800, overlap=100）
     │
     ▼
[BGE-M3 Embedding]  向量化
     │
     ▼
[Chroma]  存入向量库
     │
     ▼
──── 用户提问 ────
     │
     ▼
[BGE-M3]  问题向量化
     │
     ▼
[Chroma Retriever]  Top-K 检索（k=5）
     │
     ▼
[Prompt 组装 + 防幻觉约束]
     │
     ▼
[DeepSeek-chat]  生成回答
     │
     ▼
    输出
```

## 项目结构

```
langchain-rag-agent/
├── ingest.py           # 建库：加载 → 切分 → 向量化 → 存入 Chroma
├── query.py            # 查询：检索 → 组装 Prompt → 调用 LLM → 输出
├── requirements.txt    # 依赖
├── .env.example        # 环境变量示例
├── .gitignore
├── docs/               # 待处理的文档
└── chroma_db/          # 向量库（本地生成，不上传）
```

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/CooppperFlag/langchain-rag-agent.git
cd langchain-rag-agent
```

### 2. 创建虚拟环境并安装依赖

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env`，填入你自己的 API Key：

```env
SILICONFLOW_API_KEY=sk-xxx
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-m3

DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

### 4. 放入你的文档

把 PDF / Markdown / TXT 放进 `docs/` 目录。

### 5. 建库

```bash
python ingest.py
```

### 6. 提问

```bash
python query.py
```

## 效果示例

```
你问：RAG 的主要流程是什么？

回答：
RAG 的主要流程包括以下 6 步：
1. 文档加载
2. 文本切分
3. 向量化
4. 存入向量库
5. 检索
6. 生成

你问：今天北京天气怎么样？

回答：
资料里没有提到。
```

## 核心设计

- **防幻觉 Prompt**：明确要求模型"只能根据上下文回答，否则说'资料里没有提到'"
- **Query Rewriting**（后续扩展）：解决用户提问与文档语义距离过远的问题
- **可扩展性**：Embedding 与 LLM 均使用 OpenAI 兼容接口，可一行代码切换模型

## License

MIT
