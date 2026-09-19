import os
import re
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

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


# ===== 服务启动时执行一次 =====
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

retriever = vectordb.as_retriever(search_kwargs={"k": 5})

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

chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ===== FastAPI 应用 =====
app = FastAPI(title="LangChain RAG Agent API")


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    question: str
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    answer = chain.invoke(req.question)
    return ChatResponse(
        question=req.question,
        answer=clean_think(answer),
    )




@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def event_generator():
        async for chunk in chain.astream(req.question):
            text = clean_think(chunk)
            if text:
                yield text

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )
