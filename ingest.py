import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

# 从 .env 里读硅基流动的 Key，用来调 Embedding
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")

# 硅基流动的 OpenAI 兼容接口地址，末尾必须有 /v1
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")

# 中文向量模型，1024 维，用来把文本转成向量
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")


def load_documents():
    files = []
    files.extend(DOCS_DIR.glob("*.pdf"))
    files.extend(DOCS_DIR.glob("*.md"))
    files.extend(DOCS_DIR.glob("*.txt"))

    if not files:
        raise FileNotFoundError(f"没有在 {DOCS_DIR} 找到文档（pdf/md/txt）")

    docs = []
    for f in files:
        print(f"加载: {f.name}")
        if f.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(f))
        else:
            loader = TextLoader(str(f), encoding="utf-8")
        docs.extend(loader.load())

    print(f"共加载 {len(docs)} 个文档对象")
    return docs


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"切分成 {len(chunks)} 个 chunk")
    return chunks


def main():
    if not SILICONFLOW_API_KEY:
        raise ValueError("缺少 SILICONFLOW_API_KEY，请检查 .env")

    docs = load_documents()
    chunks = split_docs(docs)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=SILICONFLOW_API_KEY,
        base_url=SILICONFLOW_BASE_URL,
        check_embedding_ctx_length=False,
    )

    print("开始向量化并写入 Chroma...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    print(f"完成，向量库位置: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
