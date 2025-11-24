import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated

from atheon_codex import AtheonCodexClient, AtheonUnitFetchAndIntegrateModel
from bson.objectid import ObjectId
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from groq import Groq
from pydantic import BaseModel, EmailStr
from pymongo.mongo_client import MongoClient
from starlette.responses import FileResponse

__version__ = "0.3.0"


class SupportedModels(StrEnum):
    Llama3_1 = "llama3_1"
    Llama3_3 = "llama3_3"
    Deepseek_R1 = "deepseek_r1"
    OpenAIGpt_oss = "gpt_oss"


class LogEvent(StrEnum):
    CHAT_INTERACTION = "chat_interaction"
    LOGIN = "login"


class ChatRequest(BaseModel):
    model: SupportedModels = SupportedModels.Llama3_1
    message: str


class ChatResponse(BaseModel):
    response: str


class LoginRequest(BaseModel):
    email_id: EmailStr
    access_key: str


class LoginResponse(BaseModel):
    verified_id: str


class User(BaseModel):
    _id: ObjectId
    email_id: EmailStr


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_authorization_header(
    request: Request, authorization: str | None = Header(None)
) -> User:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is missing",
        )

    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme. Bearer required.",
            )

        user = request.app.mongo_db["users"].find_one({"_id": ObjectId(token)})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid verified_id",
            )

        return User(**user)

    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format",
        )


def process_groq_chat_request(
    groq_client: Groq, message: str, model: str
) -> str | None:
    return (
        groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{message}"},
            ],
            max_completion_tokens=1024,
            seed=8,
            model=model,
        )
        .choices[0]
        .message.content
    )


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    # Set API key in FastAPI app
    app.ATHEON_API_KEY = os.environ.get("ATHEON_API_KEY", "")

    app.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    app.mongo_db = MongoClient(
        os.environ.get("MONGO_URI", "mongodb+srv://localhost:27017/")
    )[os.environ.get("MONGO_DB", "arcana_hf_demo_test")]

    logging.info("Application started")

    yield

    # Clear API key to avoid leaking it
    app.ATHEON_API_KEY = ""

    app.groq_client = None

    app.mongo_db.client.close()

    logging.info("Application stopped")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health_check():
    return JSONResponse({"health_check": "pass"})


@app.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, request: Request):
    user = request.app.mongo_db["users"].find_one(
        {"email_id": payload.email_id, "access_key": payload.access_key}
    )
    if user:
        request.app.mongo_db["logs"].insert_one(
            {
                "email_id": user["email_id"],
                "timestamp": datetime.now(UTC),
                "event": LogEvent.LOGIN,
            }
        )
        verified_id = user["_id"]
        return LoginResponse(verified_id=str(verified_id))
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )


@app.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    request: Request,
    user: Annotated[User, Depends(verify_authorization_header)],
):
    logger.info(f"Received message: {payload.message}")

    client = AtheonCodexClient(api_key=request.app.ATHEON_API_KEY)

    logger.info(f"Using {payload.model}")

    inference_start_time = time.perf_counter()
    match payload.model:
        case SupportedModels.Llama3_3:
            llm_response = process_groq_chat_request(
                groq_client=request.app.groq_client,
                message=payload.message,
                model="llama-3.3-70b-versatile",
            )
        case SupportedModels.Llama3_1:
            llm_response = process_groq_chat_request(
                groq_client=request.app.groq_client,
                message=payload.message,
                model="llama-3.1-8b-instant",
            )
        case SupportedModels.Deepseek_R1:
            llm_response = process_groq_chat_request(
                groq_client=request.app.groq_client,
                message=payload.message,
                model="deepseek-r1-distill-llama-70b",
            )
        case SupportedModels.OpenAIGpt_oss:
            llm_response = process_groq_chat_request(
                groq_client=request.app.groq_client,
                message=payload.message,
                model="openai/gpt-oss-120b",
            )

    ai_response = "" if llm_response is None else llm_response.strip()
    inference_end_time = time.perf_counter()

    inference_elapsed_time = inference_end_time - inference_start_time
    logger.info(f"Inference took: {inference_elapsed_time:.4f} seconds")

    fetch_and_integrate_payload = AtheonUnitFetchAndIntegrateModel(
        query=payload.message,
        base_content=ai_response,
        include_ad_units=True,
        use_user_intent_as_filter=False,
    )

    fetch_and_integration_start_time = time.perf_counter()
    fetch_and_integration_result = client.fetch_and_integrate_atheon_unit(
        fetch_and_integrate_payload
    )
    integrated_content = fetch_and_integration_result.get("response_data", {}).get(
        "integrated_content"
    )
    fetch_and_integration_end_time = time.perf_counter()

    fetch_and_integration_elapsed_time = (
        fetch_and_integration_end_time - fetch_and_integration_start_time
    )
    logger.info(
        f"Atheon Integration took: {fetch_and_integration_elapsed_time:.4f} seconds"
    )

    request.app.mongo_db["logs"].insert_one(
        {
            "email_id": user.email_id,
            "timestamp": datetime.now(UTC),
            "event": LogEvent.CHAT_INTERACTION,
        }
    )

    return ChatResponse(response=integrated_content)


@app.get("/")
async def read_index():
    return FileResponse("./static/index.html")


@app.get("/login")
async def read_login():
    return FileResponse("./static/login.html")
