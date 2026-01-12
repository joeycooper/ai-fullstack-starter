import sqlite3
import warnings
from fastapi import FastAPI
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
import re
import time
from fastapi.routing import APIRoute
from starlette.responses import JSONResponse
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  # 导入 cosine_similarity
from dotenv import load_dotenv
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL*",
    module="urllib3",
)

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# 初始化日志记录器
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# FastAPI 应用初始化
app = FastAPI()

# CORS 中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义 DeepSeek（OpenAI 兼容）接口配置
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_EMBEDDING_MODEL = os.getenv("DEEPSEEK_EMBEDDING_MODEL", "deepseek-embedding")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))
CONNECT_TIMEOUT = float(os.getenv("CONNECT_TIMEOUT", "10"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", "0.6"))

_local_tokenizer = None
_local_model = None
_local_model_name = None


def get_env_bool(name, default="0"):
    return os.getenv(name, default) == "1"


def get_deepseek_api_key():
    key = os.getenv("DEEPSEEK_API_KEY", "")
    if key.strip().lower() in {"", "your_key_here", "your-key-here"}:
        return ""
    return key




def get_local_embedding_model():
    global _local_tokenizer, _local_model, _local_model_name
    model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "bert-base-uncased")
    if _local_tokenizer is None or _local_model is None or _local_model_name != model_name:
        _local_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _local_model = AutoModel.from_pretrained(model_name)
        _local_model.eval()
        _local_model_name = model_name
    return _local_tokenizer, _local_model

# SQLite 数据库配置
database_path = 'database.db'
conn = sqlite3.connect(database_path)
cursor = conn.cursor()

# 创建 contexts 和 paragraphs 表
cursor.execute('''CREATE TABLE IF NOT EXISTS contexts (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS paragraphs (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, embedding BLOB)''')
conn.commit()

# 确保 paragraphs 表存在 context_id 列（用于按上传批次检索）
cursor.execute("PRAGMA table_info(paragraphs)")
columns = [row[1] for row in cursor.fetchall()]
if "context_id" not in columns:
    cursor.execute("ALTER TABLE paragraphs ADD COLUMN context_id INTEGER")
    conn.commit()

# 获取最近上传的文本内容
def get_latest_context():
    cursor.execute("SELECT content FROM contexts ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    return row[0] if row else None

# 生成文本嵌入（embedding）
# 生成文本嵌入（embedding）
# 生成文本嵌入（embedding）
def generate_embedding(text):
    if not text:
        raise Exception("Text input is empty")

    deepseek_key = get_deepseek_api_key()
    use_local_fallback = get_env_bool("USE_LOCAL_EMBEDDINGS_FALLBACK")
    if not deepseek_key and not use_local_fallback:
        raise Exception("Missing DEEPSEEK_API_KEY")

    if not deepseek_key and use_local_fallback:
        tokenizer, model = get_local_embedding_model()
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).numpy()
        if len(embedding.shape) == 1:
            embedding = np.expand_dims(embedding, axis=0)
        return embedding[0]

    headers = {"Authorization": f"Bearer {deepseek_key}"}
    payload = {"model": DEEPSEEK_EMBEDDING_MODEL, "input": text}
    last_error = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = httpx.post(
                f"{DEEPSEEK_BASE_URL}/embeddings",
                json=payload,
                headers=headers,
                timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT),
            )

            if response.status_code >= 500 and attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * (attempt + 1))
                continue

            response.raise_for_status()
            data = response.json()
            embedding = np.array(data["data"][0]["embedding"], dtype=np.float32)
            if embedding.ndim != 1:
                raise Exception("Embedding shape is invalid")
            return embedding
        except httpx.HTTPStatusError as e:
            last_error = e
            if attempt < MAX_RETRIES and e.response.status_code >= 500:
                time.sleep(RETRY_BACKOFF * (attempt + 1))
                continue
            raise Exception(f"Embedding request failed: {e.response.text}")
        except httpx.TimeoutException as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * (attempt + 1))
                continue
            raise Exception("Embedding request timed out")
        except httpx.RequestError as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * (attempt + 1))
                continue
            raise Exception("Embedding request failed")

    raise Exception(f"Embedding request failed: {last_error}")


class UploadRequest(BaseModel):
    text: str


def split_paragraphs(text):
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def split_sentences(text):
    parts = re.split(r"(?<=[。！？.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def chunk_long_text(text, max_chars, overlap):
    chunks = []
    if len(text) <= max_chars:
        return [text]

    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [chunk for chunk in chunks if chunk]


def split_text_to_chunks(text, max_chunk_chars, overlap_chars):
    paragraphs = split_paragraphs(text)
    chunks = []

    for paragraph in paragraphs:
        sentences = split_sentences(paragraph) or [paragraph]
        buffer = ""
        for sentence in sentences:
            candidate = f"{buffer}{sentence}" if not buffer else f"{buffer} {sentence}"
            if len(candidate) <= max_chunk_chars:
                buffer = candidate
                continue

            if buffer:
                chunks.append(buffer.strip())
                buffer = ""

            if len(sentence) > max_chunk_chars:
                chunks.extend(chunk_long_text(sentence, max_chunk_chars, overlap_chars))
            else:
                buffer = sentence

        if buffer:
            chunks.append(buffer.strip())

    deduped = []
    seen = set()
    for chunk in chunks:
        if chunk not in seen:
            seen.add(chunk)
            deduped.append(chunk)
    return deduped


@app.post("/upload")
async def upload_context(req: UploadRequest):
    text = req.text.strip()
    if not text:
        return JSONResponse(content={"detail": "Text is empty."}, status_code=400)
    deepseek_key = get_deepseek_api_key()
    allow_no_embeddings = get_env_bool("ALLOW_NO_EMBEDDINGS")
    use_local_fallback = get_env_bool("USE_LOCAL_EMBEDDINGS_FALLBACK")
    if not deepseek_key and not (allow_no_embeddings or use_local_fallback):
        return JSONResponse(content={"detail": "缺少 DEEPSEEK_API_KEY，请先配置。"}, status_code=500)

    max_chunk_chars = int(os.getenv("CHUNK_MAX_CHARS", "480"))
    overlap_chars = int(os.getenv("CHUNK_OVERLAP_CHARS", "60"))
    paragraphs = split_text_to_chunks(text, max_chunk_chars, overlap_chars)
    if not paragraphs:
        return JSONResponse(content={"detail": "No usable paragraphs found."}, status_code=400)

    try:
        cursor.execute("INSERT INTO contexts (content) VALUES (?)", (text,))
        context_id = cursor.lastrowid

        for paragraph in paragraphs:
            if deepseek_key or use_local_fallback:
                embedding = generate_embedding(paragraph).astype(np.float32)
                embedding_blob = embedding.tobytes()
            else:
                embedding_blob = None

            cursor.execute(
                "INSERT INTO paragraphs (text, embedding, context_id) VALUES (?, ?, ?)",
                (paragraph, embedding_blob, context_id),
            )

        conn.commit()
        return JSONResponse(content={"chars": len(text), "paragraphs": len(paragraphs)})
    except Exception as e:
        logging.error(f"Upload error: {e}")
        return JSONResponse(content={"detail": "Failed to save context."}, status_code=500)

class ChatRequest(BaseModel):
    message: str
    top_k: Optional[int] = None
    return_context: Optional[bool] = None
    use_all_contexts: Optional[bool] = None


# /chat 接口：根据输入问题生成嵌入并检索相关段落作为上下文回答问题
@app.post("/chat")
async def chat(req: ChatRequest):
    deepseek_key = get_deepseek_api_key()
    allow_no_embeddings = get_env_bool("ALLOW_NO_EMBEDDINGS")
    use_local_fallback = get_env_bool("USE_LOCAL_EMBEDDINGS_FALLBACK")
    if not deepseek_key and not (allow_no_embeddings or use_local_fallback):
        return JSONResponse(content={"error": "缺少 DEEPSEEK_API_KEY，请先配置。"}, status_code=500)
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant. Answer concisely based on the provided context."},
            {"role": "user", "content": req.message},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }

    try:
        use_all_contexts = (
            req.use_all_contexts
            if req.use_all_contexts is not None
            else get_env_bool("USE_ALL_CONTEXTS", "1")
        )

        # 计算用户输入问题的嵌入，确保是 2D 数组
        user_message_embedding = generate_embedding(req.message).reshape(1, -1)  # 确保是 2D 数组
        logging.debug(f"User message embedding shape: {user_message_embedding.shape}")  # 调试输出
        
        # 检索所有段落
        if use_all_contexts:
            cursor.execute("SELECT text, embedding, context_id FROM paragraphs")
        else:
            cursor.execute("SELECT id FROM contexts ORDER BY id DESC LIMIT 1")
            latest = cursor.fetchone()
            if not latest:
                return JSONResponse(content={"error": "未找到已上传的上下文，请先上传内容。"}, status_code=400)
            cursor.execute(
                "SELECT text, embedding, context_id FROM paragraphs WHERE context_id = ?",
                (latest[0],),
            )
        all_paragraphs = cursor.fetchall()
        
        if not all_paragraphs:
            return JSONResponse(content={"error": "未找到已上传的上下文，请先上传内容。"}, status_code=400)

        similarities = []
        for paragraph in all_paragraphs:
            # 确保段落嵌入是二维数组
            if paragraph[1] is None:
                return JSONResponse(
                    content={"error": "未生成向量，无法检索。请配置 DEEPSEEK_API_KEY 并重新上传。"},
                    status_code=400,
                )
            paragraph_embedding = np.frombuffer(paragraph[1], dtype=np.float32).reshape(1, -1)
            logging.debug(f"Paragraph embedding shape: {paragraph_embedding.shape}")  # 调试输出
            similarity = cosine_similarity(user_message_embedding, paragraph_embedding)
            similarities.append((paragraph[0], similarity[0][0], paragraph[2]))

        # 找到最相关的段落（Top-K）
        similarities.sort(key=lambda x: x[1], reverse=True)
        default_top_k = int(os.getenv("TOP_K", "3"))
        top_k = req.top_k if req.top_k and req.top_k > 0 else default_top_k
        max_context_chars = int(os.getenv("MAX_CONTEXT_CHARS", "1400"))

        top_paragraphs = []
        total_chars = 0
        for text, score, context_id in similarities[:top_k]:
            if total_chars + len(text) > max_context_chars and top_paragraphs:
                break
            top_paragraphs.append((text, score, context_id))
            total_chars += len(text)

        # 将检索到的段落拼接到 system prompt 中
        payload["messages"][0]["content"] += "\nRelevant context:\n" + "\n\n".join(
            [item[0] for item in top_paragraphs]
        )
        
        # 发送请求到 Ollama 模型并返回回答
        if not deepseek_key:
            fallback_answer = "未配置 DEEPSEEK_API_KEY，暂不调用模型。以下为检索到的相关内容：\n\n"
            fallback_answer += "\n\n".join([item[0] for item in top_paragraphs])
            return JSONResponse(
                content={
                    "answer": fallback_answer,
                    "retrieval": {
                        "top_k": top_k,
                        "chunks": [
                            {"text": text, "score": float(score), "context_id": context_id}
                            for text, score, context_id in top_paragraphs
                        ],
                    },
                }
            )

        headers = {"Authorization": f"Bearer {deepseek_key}"}
        chat_response = None
        last_error = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = httpx.post(
                    f"{DEEPSEEK_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT),
                )

                if response.status_code >= 500 and attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF * (attempt + 1))
                    continue

                response.raise_for_status()
                chat_response = response.json()
                break
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                detail = e.response.text
                logging.error(f"Upstream error: {status_code} {detail}")
                return JSONResponse(
                    content={"error": "上游服务返回错误。", "status": status_code},
                    status_code=502,
                )
            except httpx.TimeoutException as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF * (attempt + 1))
                    continue
                return JSONResponse(
                    content={"error": "请求模型超时，请稍后再试。"},
                    status_code=504,
                )
            except httpx.RequestError as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF * (attempt + 1))
                    continue
                return JSONResponse(
                    content={"error": "请求模型失败，请检查网络或上游服务。"},
                    status_code=502,
                )

        if chat_response is None:
            logging.error(f"Request error: {last_error}")
            return JSONResponse(
                content={"error": "请求模型失败，请稍后再试。"},
                status_code=502,
            )
        if req.return_context:
            chat_response["retrieval"] = {
                "top_k": top_k,
                    "chunks": [
                        {"text": text, "score": float(score), "context_id": context_id}
                        for text, score, context_id in top_paragraphs
                    ],
                }

        return JSONResponse(content=chat_response)

    except httpx.RequestError as e:
        logging.error(f"Request error: {e}")
        return JSONResponse(
            content={"error": "请求模型超时或不可达，请确认 Ollama 服务已启动。"},
            status_code=504,
        )
    except Exception as e:
        logging.exception("Unexpected error")
        return JSONResponse(content={"error": f"服务端异常：{e}"}, status_code=500)
