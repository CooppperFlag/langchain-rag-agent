import os
import re
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

CHROMA_DIR = BASE_DIR / "chroma_db"

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")


def clean_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


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
#retriever 是检索器。向量库（Chroma）通过 .as_retriever() 包装成一个“能根据问题返回相关文档”的对象。你给它一个问题，它返回一批相关的 chunk。
#{"k"：5}每次检索，返回最相似的 5 个 chunk
#kwargs keyword arguements 关键字参数
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0,
    )
#deepseek是llm 硅基流动的BGM-M3是Embedding embedding把文本转成向量（数字）用于检索 llm读上下文、生成回答，用于生成
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
#question:用户输入的问题 context 从知识库检索出来的chunk

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
# | 管道符 把数据从上一步喂到下一步的传送带
    print("RAG Agent 已就绪，输入 q 退出。")
    while True:
        question = input("\n你问：").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break
        if not question:
            continue

        answer = chain.invoke(question)
        print("\n回答：")
        print(clean_think(answer))


if __name__ == "__main__":
    main()
