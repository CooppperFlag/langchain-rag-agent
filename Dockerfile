# 1. 基础镜像：官方 Python 3.12 精简版
FROM python:3.12-slim

# 2. 设置工作目录（容器里的 /app）
WORKDIR /app

# 3. 先 COPY 依赖清单（利用分层缓存）
COPY requirements.txt .

# 4. 装依赖：用清华镜像加速，不缓存 pip 文件
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 再 COPY 会变的代码
COPY api.py ingest.py query.py ./

# 6. 拷贝文档目录
COPY docs/ ./docs/

# 7. 声明容器监听 8000 端口（只是个注释性说明，不实际开端口）
EXPOSE 8000

# 8. 容器启动时执行的命令
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
