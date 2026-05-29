from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import Cookie, Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.corrosion.auth import Role, SessionStore, UserRecord, UserStore
from src.corrosion.config import load_config
from src.corrosion.inference import CorrosionAnalyzer
from src.corrosion.schema import PredictionResponse

ROOT_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT_DIR / "api" / "static"
UPLOAD_DIR = ROOT_DIR / "outputs" / "uploads"
PREDICTION_DIR = ROOT_DIR / "outputs" / "predictions"
USER_STORE_PATH = ROOT_DIR / "data" / "users.json"
CONFIG_PATH = Path(os.getenv("APP_CONFIG", ROOT_DIR / "configs" / "app.yaml"))
MODEL_PATH = os.getenv("MODEL_PATH")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
SESSION_COOKIE_NAME = "corrosion_session"

app = FastAPI(title="Ship Hull Corrosion Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(ROOT_DIR / "outputs")), name="outputs")

config = load_config(CONFIG_PATH)
analyzer: CorrosionAnalyzer | None = None
users = UserStore(USER_STORE_PATH)
sessions = SessionStore()


class LoginRequest(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: Role


def require_user(corrosion_session: str | None = Cookie(default=None)) -> UserRecord:
    user = sessions.get(corrosion_session)
    if user is None:
        raise HTTPException(status_code=401, detail="Login diperlukan.")
    return user


def require_admin(current_user: UserRecord = Depends(require_user)) -> UserRecord:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Akses admin diperlukan.")
    return current_user


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "login.html")


@app.get("/app", response_model=None)
def dashboard(corrosion_session: str | None = Cookie(default=None)):
    if sessions.get(corrosion_session) is None:
        return RedirectResponse("/", status_code=303)
    return FileResponse(STATIC_DIR / "app.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    model_path = Path(MODEL_PATH or config.model.path)
    return {
        "status": "ok",
        "model_path": str(model_path),
        "model_available": str(model_path.exists()).lower(),
    }


@app.post("/api/login")
def login(payload: LoginRequest, response: Response) -> dict[str, str]:
    user = users.authenticate(payload.username, payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Username atau password salah.")

    token = sessions.create(user)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=60 * 60 * 8,
    )
    return {"username": user.username, "role": user.role}


@app.post("/api/logout")
def logout(
    response: Response,
    corrosion_session: str | None = Cookie(default=None),
) -> dict[str, str]:
    sessions.delete(corrosion_session)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"status": "logged_out"}


@app.get("/logout", response_model=None)
def logout_page(
    corrosion_session: str | None = Cookie(default=None),
):
    sessions.delete(corrosion_session)
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return response


@app.get("/api/me")
def me(current_user: UserRecord = Depends(require_user)) -> dict[str, str]:
    return {"username": current_user.username, "role": current_user.role}


@app.get("/api/users")
def list_users(current_user: UserRecord = Depends(require_admin)) -> list[dict[str, str]]:
    return [{"username": user.username, "role": user.role} for user in users.list_users()]


@app.post("/api/users")
def create_user(
    payload: CreateUserRequest,
    current_user: UserRecord = Depends(require_admin),
) -> dict[str, str]:
    try:
        user = users.add_user(payload.username, payload.password, payload.role)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"username": user.username, "role": user.role}


@app.delete("/api/users/{username}")
def delete_user(
    username: str,
    current_user: UserRecord = Depends(require_admin),
) -> dict[str, str]:
    if username == current_user.username:
        raise HTTPException(status_code=400, detail="Admin tidak bisa menghapus akun yang sedang dipakai.")
    try:
        users.delete_user(username)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "deleted"}


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    mm_per_pixel: float | None = Form(default=None),
    current_user: UserRecord = Depends(require_user),
) -> PredictionResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Only JPEG, PNG, and WEBP images are supported.")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded image is too large.")

    image_array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    extension = _safe_extension(file.filename or "upload.jpg")
    upload_path = UPLOAD_DIR / f"{uuid4().hex}{extension}"
    cv2.imwrite(str(upload_path), image)

    try:
        active_analyzer = get_analyzer()
        return active_analyzer.analyze_image(
            image_path=upload_path,
            output_dir=PREDICTION_DIR,
            mm_per_pixel=mm_per_pixel,
            artifact_url_prefix="/outputs/predictions",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _safe_extension(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"


def get_analyzer() -> CorrosionAnalyzer:
    global analyzer
    if analyzer is not None:
        return analyzer

    model_path = Path(MODEL_PATH or config.model.path)
    if not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model file is not available: {model_path}. Put trained weights at this path.",
        )

    analyzer = CorrosionAnalyzer(config=config, model_path=model_path)
    return analyzer
