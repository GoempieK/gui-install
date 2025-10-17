"""Example FastAPI bridge that lets a Passivbot instance register with the central backend.

Copy or adapt this file inside your Passivbot runtime image. Extend the `command_endpoint`
handler so that it invokes real trading, backtest, or optimisation routines.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from shlex import split as shlex_split
from typing import Any, Dict, Literal

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator, model_validator

from agent.process_manager import BotProcessManager

BOT_ID = os.environ.get("BOT_ID", "unknown-bot")
BOT_ROLE = os.environ.get("BOT_ROLE", "worker")
BOT_API_KEY = os.environ["BOT_API_KEY"]
BACKEND_URL = os.environ.get("BACKEND_URL")
SELF_URL = os.environ.get("SELF_URL", "http://localhost:9000")
BOT_IP = os.environ.get("BOT_IP")
BOT_VERSION = os.environ.get("BOT_VERSION")
AGENT_ROOT = Path(os.environ.get("AGENT_ROOT", Path(__file__).resolve().parent))
LOGS_DIR = Path(os.environ.get("AGENT_LOGS_DIR", AGENT_ROOT / "logs"))
PASSIVBOT_ROOT = Path(os.environ.get("PASSIVBOT_ROOT", AGENT_ROOT / "passivbot"))
PASSIVBOT_CONFIGS_DIR = Path(
    os.environ.get("PASSIVBOT_CONFIGS_DIR", PASSIVBOT_ROOT / "configs")
)
MANAGED_BOTS_ROOT = Path(
    os.environ.get("MANAGED_BOTS_ROOT", PASSIVBOT_ROOT.parent)
)
DEFAULT_BASE_COMMAND = tuple(
    arg for arg in shlex_split(os.environ.get("PASSIVBOT_BASE_COMMAND", "")) if arg
)

app = FastAPI(title=f"Passivbot Agent {BOT_ID}")
process_manager = BotProcessManager(logs_dir=LOGS_DIR)


class CommandRequest(BaseModel):
    command: Literal["start_trade", "start_backtest", "stop", "custom"]
    payload: dict = Field(default_factory=dict)


class StatusPayload(BaseModel):
    id: str
    role: str
    heartbeat_ts: datetime
    state: Literal["idle", "running", "error"]
    current_task: str | None = None
    details: dict = Field(default_factory=dict)


class StartBotRequest(BaseModel):
    bot_id: str
    command: list[str] | None = None
    config_path: str | None = Field(
        default=None,
        description="Path to configuration file. Relative paths resolve against PASSIVBOT_CONFIGS_DIR.",
    )
    workdir: str | None = Field(
        default=None,
        description="Working directory for the process."
    )
    env: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @validator("bot_id")
    def _validate_bot_id(cls, value: str) -> str:
        if not value or value.strip() == "":
            raise ValueError("bot_id is required")
        return value

    @model_validator(mode="after")
    def _ensure_command_or_default(self):
        if self.command:
            return self
        if self.config_path or DEFAULT_BASE_COMMAND:
            return self
        raise ValueError("Either command or PASSIVBOT_BASE_COMMAND must be provided")


class StopBotRequest(BaseModel):
    bot_id: str
    force: bool = Field(
        default=False,
        description="If true, sends SIGKILL after attempting graceful shutdown.",
    )


class StartStopResponse(BaseModel):
    detail: str
    state: dict[str, Any]


class MetadataUpdateRequest(BaseModel):
    bot_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @validator("bot_id")
    def _validate_metadata_bot(cls, value: str) -> str:
        if not value or value.strip() == "":
            raise ValueError("bot_id is required")
        return value


class ConfigUploadRequest(BaseModel):
    filename: str = Field(..., description="Filename to store under the bot's config directory")
    content: Dict[str, Any] = Field(default_factory=dict, description="Passivbot configuration payload")

    @validator("filename")
    def _valid_filename(cls, value: str) -> str:
        candidate = value.strip()
        if not candidate:
            raise ValueError("filename is required")
        return candidate


class ConfigUploadResponse(BaseModel):
    detail: str
    path: str
    filename: str


class ConfigDownloadResponse(BaseModel):
    filename: str
    content: Dict[str, Any]


_status = StatusPayload(
    id=BOT_ID,
    role=BOT_ROLE,
    heartbeat_ts=datetime.now(timezone.utc),
    state="idle",
)


async def authenticate(x_api_key: str = Header(...)) -> None:
    if x_api_key != BOT_API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


@app.get("/status", response_model=StatusPayload)
async def status_endpoint(_: None = Depends(authenticate)) -> StatusPayload:
    return _status


@app.post("/command")
async def command_endpoint(request: CommandRequest, _: None = Depends(authenticate)) -> JSONResponse:
    global _status
    _status = _status.copy(update={"state": "running", "current_task": request.command})

    # TODO: replace with the actual Passivbot command runner for your environment.
    await asyncio.sleep(0.1)
    if request.command == "stop":
        _status = _status.copy(update={"state": "idle", "current_task": None})

    return JSONResponse({"detail": f"Command {request.command} accepted"})


def _resolve_config_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = PASSIVBOT_CONFIGS_DIR / candidate
    return candidate


def _resolve_workdir(raw_path: str | None) -> Path | None:
    if not raw_path:
        return PASSIVBOT_ROOT if PASSIVBOT_ROOT.exists() else None
    path = Path(raw_path)
    return path


def _sanitize_identifier(raw: str) -> str:
    cleaned = "".join(ch for ch in raw if ch.isalnum() or ch in {"-", "_"})
    if not cleaned:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid identifier")
    return cleaned


def _bot_workspace(bot_id: str) -> Path:
    normalized = _sanitize_identifier(bot_id)
    return MANAGED_BOTS_ROOT / normalized


def _bot_config_dir(bot_id: str) -> Path:
    workspace = _bot_workspace(bot_id)
    path = workspace / "configs"
    path.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve()
    root = MANAGED_BOTS_ROOT.resolve()
    if root not in resolved.parents and resolved != root:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Config path escapes managed root")
    return resolved


def _config_file_path(bot_id: str, filename: str) -> Path:
    config_dir = _bot_config_dir(bot_id)
    sanitized_name = Path(filename).name
    if not sanitized_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename")
    target = (config_dir / sanitized_name).resolve()
    if config_dir not in target.parents and target != config_dir:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Config path escapes managed root")
    return target


@app.post("/command/start", response_model=StartStopResponse)
async def start_bot(
    request: StartBotRequest,
    _: None = Depends(authenticate),
) -> StartStopResponse:
    command = list(request.command) if request.command else list(DEFAULT_BASE_COMMAND)
    if not command:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No command provided")

    config_path = _resolve_config_path(request.config_path)
    if config_path and not config_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Config file not found: {config_path}",
        )

    if request.command is None and config_path:
        command = command + [str(config_path)]

    workdir = _resolve_workdir(request.workdir)

    try:
        state = await process_manager.start_bot(
            bot_id=request.bot_id,
            command=command,
            config_path=config_path,
            workdir=workdir,
            env=request.env,
            metadata=request.metadata,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    return StartStopResponse(detail="Bot process started", state=state.to_payload())


@app.post("/command/stop", response_model=StartStopResponse)
async def stop_bot(
    request: StopBotRequest,
    _: None = Depends(authenticate),
) -> StartStopResponse:
    try:
        if request.force:
            state = await process_manager.kill_bot(request.bot_id)
        else:
            state = await process_manager.stop_bot(request.bot_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    detail = "Bot process stopped" if state.exit_code == 0 else "Bot process terminated"
    return StartStopResponse(detail=detail, state=state.to_payload())


@app.get("/command/status")
async def list_managed_bots(_: None = Depends(authenticate)) -> dict[str, Any]:
    return {"bots": process_manager.snapshot()}


@app.post("/command/metadata", response_model=StartStopResponse)
async def update_bot_metadata(
    request: MetadataUpdateRequest,
    _: None = Depends(authenticate),
) -> StartStopResponse:
    try:
        state = await process_manager.update_metadata(request.bot_id, request.metadata)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    return StartStopResponse(detail="Metadata updated", state=state.to_payload())


@app.put("/configs/{bot_id}", response_model=ConfigUploadResponse)
async def upload_config(
    bot_id: str,
    request: ConfigUploadRequest,
    _: None = Depends(authenticate),
) -> ConfigUploadResponse:
    target = _config_file_path(bot_id, request.filename)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(request.content, indent=2, sort_keys=True))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    return ConfigUploadResponse(detail="Config stored", path=str(target), filename=target.name)


@app.get("/configs/{bot_id}", response_model=ConfigDownloadResponse)
async def fetch_config(
    bot_id: str,
    filename: str = Query(..., description="Filename previously uploaded for the bot"),
    _: None = Depends(authenticate),
) -> ConfigDownloadResponse:
    target = _config_file_path(bot_id, filename)
    if not target.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Config not found")
    try:
        content = json.loads(target.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Invalid JSON: {exc}") from exc
    return ConfigDownloadResponse(filename=target.name, content=content)


async def heartbeat_task() -> None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            await asyncio.sleep(15)
            if not BACKEND_URL:
                continue
            global _status
            _status = _status.copy(update={"heartbeat_ts": datetime.now(timezone.utc)})
            managed_bots = process_manager.snapshot()
            payload = {
                "status": _status.state,
                "version": BOT_VERSION,
                "metadata": {
                    "current_task": _status.current_task,
                    "details": _status.details,
                    "managed_bots": managed_bots,
                },
            }
            try:
                await client.post(
                    f"{BACKEND_URL.rstrip('/')}/bots/{BOT_ID}/heartbeat",
                    json=payload,
                    headers={"x-bot-token": BOT_API_KEY},
                )
            except httpx.HTTPError:
                pass  # log in production


@app.on_event("startup")
async def on_startup() -> None:
    if not BACKEND_URL:
        return
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            await client.post(
                f"{BACKEND_URL.rstrip('/')}/bots/register",
                json={
                    "id": BOT_ID,
                    "role": BOT_ROLE,
                    "hostname": SELF_URL,
                    "ip": BOT_IP,
                    "version": BOT_VERSION,
                    "metadata": {},
                },
                headers={"x-bot-token": BOT_API_KEY},
            )
        except httpx.HTTPError:
            pass
    asyncio.create_task(heartbeat_task())


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await process_manager.shutdown()
