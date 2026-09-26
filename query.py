import os
import re
import requests
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

CHROMA_DIR = BASE_DIR / "chroma_db"

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

SILICONFLOW_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
SILICONFLOW_RERANK_URL = "https://api.siliconflow.cn/v1/rerank"


def clean_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def rerank_documents(query: str, docs, top_n: int = 5):
    """调用硅基流动 Rerank API 对文档重排序，返回 top_n 个。"""
    if not docs:
        return docs

    documents = [doc.page_content for doc in docs]

    payload = {
        "model": SILICONFLOW_RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": top_n,
        "return_documents": False,
    }

    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }

    resp = requests.post(
        SILICONFLOW_RERANK_URL,
        json=payload,
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json()["results"]

    results = sorted(results, key=lambda x: x["relevance_score"], reverse=True)

    reranked = [docs[r["index"]] for r in results[:top_n]]

    print(f"[Rerank] 输入 {len(docs)} 个候选，返回 {len(reranked)} 个")
    for i, r in enumerate(results[:top_n]):
        print(f"  Top{i+1}: score={r['relevance_score']:.4f} | {documents[r['index']][:60]}...")

    return reranked


def main():
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=SILICONFLOW_API_KEY,
        base_url=SILICONFLOW_BASE_URL,
        check_embedding_ctx_length=False,
    )

    vectordb = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 20})

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是一个严谨的 RAG 助手。"
            "只能根据下面提供的上下文回答问题。"
            "如果上下文里没有答案，就直接说“资料里没有提到”。"
            "严禁编造。\n\n"
            "上下文：\n{context}",
        ),
        ("human", "{question}"),
    ])

    rewrite_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是一个查询改写助手。"
            "用户的问题可能口语化、简短、含有指代词。"
            "请把用户问题改写成更适合在知识库中检索的查询语句。"
            "要求：\n"
            "1. 保留核心关键词\n"
            "2. 补充同义词或相关术语\n"
            "3. 去掉口语化语气词、指代词\n"
            "4. 只输出改写后的查询语句，不要任何解释、不要引号\n\n"
            "示例：\n"
            "输入：那个写大模型应用的框架是啥来着\n"
            "输出：LangChain 大模型应用 开发框架 Python\n\n"
            "输入：怎么防止模型乱说\n"
            "输出：大模型 幻觉 防止 编造 方法",
        ),
        ("human", "{question}"),
    ])

    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    def retrieve_with_rewrite(question: str) -> str:
        """先用 LLM 改写问题，再用改写后的问题去检索 + Rerank 精排。"""
        rewritten = rewrite_chain.invoke({"question": question})
        rewritten = rewritten.strip()
        print(f"\n[改写前] {question}")
        print(f"[改写后] {rewritten}")
        docs = retriever.invoke(rewritten)
        print(f"[检索] 初筛得到 {len(docs)} 个候选")
        docs = rerank_documents(rewritten, docs, top_n=5)
        return format_docs(docs)

    chain = (
        {
            "context": RunnableLambda(retrieve_with_rewrite),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG Agent 已就绪，输入 q 退出。")
    while True:
        question = input("\n你问：").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break
        if not question:
            continue

        safe_question = question.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

        try:
            answer = chain.invoke(safe_question)
            print("\n回答：")
            print(clean_think(answer))
        except Exception as e:
            print(f"\n[错误] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
