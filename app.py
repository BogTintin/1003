# back.py
import os
import time
import logging
import traceback
from typing import List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

from openai import OpenAI

# ---- logging ----
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("app")

# ---- config ----
MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "800"))  # demo 預設稍微給大一些
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT")  # 可為 None

if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY as an environment variable")

# 重要：gpt-4o-* 走 Responses API；project 可選但若有就帶上
client = OpenAI(
    api_key=OPENAI_API_KEY,
    project=OPENAI_PROJECT if OPENAI_PROJECT else None,
)

# 啟動時印出精簡組態（避免洩漏完整金鑰）
masked_key = OPENAI_API_KEY[:8] + "..." if OPENAI_API_KEY else "None"
masked_proj = (OPENAI_PROJECT[:8] + "...") if OPENAI_PROJECT else "None"
log.info(f"Model={MODEL}  MAX_TOKENS={MAX_TOKENS}  TEMP={TEMPERATURE}")
log.info(f"OPENAI_API_KEY(prefix)={masked_key}  OPENAI_PROJECT(prefix)={masked_proj}")

app = FastAPI()
START_TS = time.time()

# CORS：先全部放行方便測試；之後可改白名單
allow_origins = os.getenv("CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allow_origins.split(",")] if allow_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Schemas ----------
class ChatMessage(BaseModel):
    role: str = Field(..., description="user/assistant/system")
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

# ---------- Endpoints ----------
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
      <body>
        <h2>Insurance Agent Backend (Demo)</h2>
        <ul>
          <li>GET <code>/health</code></li>
          <li>GET <code>/version</code></li>
          <li>POST <code>/chat</code> with {"messages":[{"role":"user","content":"hi"}]}</li>
        </ul>
      </body>
    </html>
    """

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "history_limit": 8,
        "cors": [allow_origins] if isinstance(allow_origins, str) else allow_origins,
        "project_set": bool(OPENAI_PROJECT),
        "api_key_set": bool(OPENAI_API_KEY),
        "uptime_sec": int(time.time() - START_TS),
    }

@app.post("/chat")
def chat(req: ChatRequest, request: Request):
    """
    使用 Responses API 呼叫 gpt-4o-mini
    把聊天訊息串成單一文字輸入（Demo 夠用）
    """
    try:
        if not req.messages:
            return JSONResponse(status_code=400, content={"detail": "messages 不可為空"})

        # 將 messages 簡單串接（也可以自己定更好的格式）
        parts = []
        for m in req.messages:
            r = (m.role or "user").strip().lower()
            if r not in ("user", "system", "assistant"):
                r = "user"
            parts.append(f"{r.upper()}: {m.content}")
        prompt = "\n".join(parts) + "\nASSISTANT:"

        resp = client.responses.create(
            model=MODEL,
            input=prompt,
            max_output_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        # 簡單取得輸出文字（OpenAI SDK 1.x 提供 output_text）
        reply_text = getattr(resp, "output_text", None)
        if not reply_text:
            # 作為保險，若 SDK 未提供屬性時嘗試從候選結構中取出
            try:
                reply_text = resp.output[0].content[0].text
            except Exception:
                reply_text = ""

        return {"reply": reply_text}

    except Exception as e:
        # 詳細錯誤打到 logs，且把可讀性好的訊息回傳
        log.error("=== /chat upstream error ===")
        log.error("Path: %s", request.url.path)
        log.error("Client: %s", request.client)
        log.error("Error: %s: %s", type(e).__name__, str(e))
        traceback.print_exc()

        return JSONResponse(
            status_code=502,
            content={
                "detail": "upstream_error",
                "error": f"{type(e).__name__}: {str(e)}",
                "hint": "檢查 OPENAI_API_KEY / OPENAI_PROJECT / MODEL_NAME 是否正確，或稍後重試",
            },
        )