# RAG 与 LangChain 简介

## 什么是 RAG

RAG 全称 Retrieval-Augmented Generation，中文叫检索增强生成。
它的核心思想是：在让大模型回答问题之前，先从知识库中检索出相关文档，
把文档作为上下文一起送给大模型，让大模型基于这些真实材料回答。

RAG 能有效减少大模型的幻觉问题。

## RAG 的主要流程

1. 文档加载：把 PDF、Markdown、网页等加载为文本。
2. 文本切分：把长文档切成小段落，称为 chunk。
3. 向量化：用 Embedding 模型把每个 chunk 转成向量。
4. 存入向量库：把向量存入 Chroma 或 FAISS 等向量数据库。
5. 检索：用户提问时，把问题也向量化，找出最相似的 chunk。
6. 生成：把检索到的 chunk 和问题一起送给 LLM，生成回答。

## LangChain 是什么

LangChain 是一个用于构建大模型应用的 Python 框架。
它提供了文档加载器、文本切分器、向量库封装、Prompt 模板、
LLM 调用封装等一系列组件，让 RAG 系统的开发变得简单。

## 常用向量数据库

- Chroma：轻量、本地、易上手，适合入门。
- FAISS：Facebook 开源，速度快，适合大规模。
- Milvus：分布式，适合生产环境。
- Qdrant：Rust 编写，性能好。
