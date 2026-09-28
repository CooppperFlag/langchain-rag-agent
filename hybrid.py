"""Hybrid Search：向量检索 + BM25 关键词检索，RRF 融合。

要点（面试可讲）：
1. BM25 是内存索引，进程重启就没了，必须在服务启动时从 docs/ 重建。
2. BM25 的 chunk 必须和 ingest.py 里进 Chroma 的 chunk 保持一致
   （同样的 loader、同样的 chunk_size=200、chunk_overlap=100）。
3. 中文必须用 jieba 分词，BM25 默认按空格切，中文会直接废掉。
4. EnsembleRetriever 内部用的是 weighted RRF，只看排名不看分数。
"""

import jieba
from ingest import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

# LangChain 1.x：经典 retriever 挪到 langchain_classic
from langchain_classic.retrievers import EnsembleRetriever


def _load_and_split(docs_dir: Path):
    """从 docs_dir 加载 PDF/MD/TXT 并切分，参数与 ingest.py 一致。"""
    raw_docs = []
    for p in sorted(docs_dir.rglob("*")):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(p))
            elif suffix in {".md", ".txt"}:
                loader = TextLoader(str(p), encoding="utf-8")
            else:
                continue
            raw_docs.extend(loader.load())
            print(f"[Hybrid] 已加载 {p.name}")
        except Exception as e:
            print(f"[Hybrid] 跳过 {p.name}: {e}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"[Hybrid] 原始文档 {len(raw_docs)} 个 → chunk {len(chunks)} 个")
    return chunks


def build_bm25_retriever(docs_dir: Path, k: int = 20):
    """构建中文 BM25 检索器。"""
    chunks = _load_and_split(docs_dir)
    if not chunks:
        raise RuntimeError(f"docs/ 里没找到可用文档：{docs_dir}")

    retriever = BM25Retriever.from_documents(
        chunks,
        preprocess_func=jieba.lcut,
        stop_words=[],
    )
    retriever.k = k
    print(f"[BM25] 索引构建完成，共 {len(chunks)} chunk，k={k}")
    return retriever


def build_hybrid_retriever(vector_retriever, docs_dir: Path, k: int = 20, weights=(0.5, 0.5)):
    """向量检索 + BM25，用 EnsembleRetriever（内部 weighted RRF）融合。"""
    bm25_retriever = build_bm25_retriever(docs_dir, k=k)
    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=list(weights),
    )
    print(f"[Hybrid] EnsembleRetriever 就绪，weights={weights}")
    return ensemble
