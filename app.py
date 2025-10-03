import os, time, logging, traceback
from typing import List
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

from openai import OpenAI
import httpx

# -------- config & logging --------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("app")

MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "800"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

# 重要：strip 去不可見字元；避免 Render 上貼入多餘空白/換行
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_PROJECT = (os.getenv("OPENAI_PROJECT") or "").strip() or None

if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY")

# 啟動時只記錄遮罩
def mask(s: str, n=4):
    return (s[:n] + "..." + s[-n:]) if s and len(s) > n*2 else ("..." if s else "None")

log.info(f"Model={MODEL}  MAX_TOKENS={MAX_TOKENS}  TEMP={TEMPERATURE}")
log.info(f"OPENAI_API_KEY(prefix)={mask(OPENAI_API_KEY)}  OPENAI_PROJECT={mask(OPENAI_PROJECT or '')}")

# 不在 import 當下就強綁，改為惰性建立，避免初始化期炸掉
_client: OpenAI | None = None
def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT)
    return _client

app = FastAPI()
START_TS = time.time()

allow_origins = os.getenv("CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allow_origins.split(",")] if allow_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- schemas --------
class ChatMessage(BaseModel):
    role: str = Field(..., description="user/assistant/system")
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

# -------- pages --------
@app.get("/", response_class=HTMLResponse)
def root():
    return """<h3>Insurance Agent Backend</h3><ul>
    <li>GET /health</li><li>GET /version</li><li>GET /debug/key</li><li>GET /debug/ping</li>
    <li>POST /chat</li></ul>"""

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

# ---- 便於 Render 側排查的 debug 端點 ----
@app.get("/debug/key")
def debug_key():
    return {
        "prefix": OPENAI_API_KEY[:6],
        "suffix": OPENAI_API_KEY[-6:],
        "length": len(OPENAI_API_KEY),
        "project_prefix": (OPENAI_PROJECT[:6] if OPENAI_PROJECT else None),
    }

@app.get("/debug/ping")
def debug_ping():
    # 直接以 httpx 打 OpenAI /models，用目前雲端的 KEY 驗證是否被接受
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    try:
        r = httpx.get("https://api.openai.com/v1/models", headers=headers, timeout=20)
        return {"status_code": r.status_code, "ok": r.status_code == 200, "body_head": r.text[:200]}
    except Exception as e:
        return JSONResponse(status_code=502, content={"detail": f"ping_failed: {type(e).__name__}: {e}"})

# -------- chat with fallback --------
@app.post("/chat")
def chat(req: ChatRequest, request: Request):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages 不可為空")

    # 將 messages 串成 prompt（demo 用）
    parts = []
    for m in req.messages:
        r = (m.role or "user").strip().lower()
        if r not in ("user", "system", "assistant"):
            r = "user"
        parts.append(f"{r.upper()}: {m.content}")
    prompt = "\n".join(parts) + "\nASSISTANT:"

    client = get_client()

    # 先試 Responses API
    try:
        resp = client.responses.create(
            model=MODEL,
            input=prompt,
            max_output_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        text = getattr(resp, "output_text", None)
        if not text:
            try:
                text = resp.output[0].content[0].text
            except Exception:
                text = ""
        if text.strip():
            return {"reply": text}
        # 沒內容則進入 fallback
        raise RuntimeError("empty_response_from_responses_api")
    except Exception as e1:
        # 若是認證錯或 401，或 Responses 不支援，fallback 到 Chat Completions
        try:
            log.warning("Responses API failed: %s: %s; fallback to Chat Completions", type(e1).__name__, e1)
            cc = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            text = cc.choices[0].message.content
            return {"reply": text}
        except Exception as e2:
            log.error("=== /chat upstream error ===")
            log.error("Path: %s", request.url.path)
            log.error("Client: %s", request.client)
            log.error("Errors: %s / %s", f"{type(e1).__name__}: {e1}", f"{type(e2).__name__}: {e2}")
            traceback.print_exc()
            return JSONResponse(
                status_code=502,
                content={
                    "detail": "upstream_error",
                    "error": f"{type(e1).__name__}: {e1}  |  {type(e2).__name__}: {e2}",
                    "hint": "先打 /debug/ping 檢查 Render 上的 KEY 是否被 OpenAI 接受；若 Responses 401 改走 Chat Completions。",
                },
            )