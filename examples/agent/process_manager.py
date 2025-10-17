from __future__ import annotations

import asyncio
import contextlib
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Optional


@dataclass
class BotProcessMetadata:
  bot_id: str
  command: tuple[str, ...]
  config_path: Path | None
  workdir: Path | None
  env: Dict[str, str] = field(default_factory=dict)


@dataclass
class BotProcessState:
  bot_id: str
  status: str = "idle"
  started_at: datetime | None = None
  finished_at: datetime | None = None
  exit_code: int | None = None
  last_error: str | None = None
  log_path: Path | None = None
  log_tail: Deque[str] = field(default_factory=lambda: deque(maxlen=200))
  metadata: Dict[str, Any] = field(default_factory=dict)

  def to_payload(self) -> Dict[str, Any]:
    config_path = self.metadata.get("config_path")
    config_name = self.metadata.get("config_name")
    wallet_exposure = self.metadata.get("wallet_exposure")
    return {
      "bot_id": self.bot_id,
      "status": self.status,
      "started_at": self.started_at.isoformat() if self.started_at else None,
      "finished_at": self.finished_at.isoformat() if self.finished_at else None,
      "uptime_seconds": self.uptime_seconds,
      "exit_code": self.exit_code,
      "last_error": self.last_error,
      "log_path": str(self.log_path) if self.log_path else None,
      "log_tail": list(self.log_tail),
      "config_path": config_path,
      "config_name": config_name,
      "wallet_exposure": wallet_exposure,
      "metadata": self.metadata,
    }

  @property
  def uptime_seconds(self) -> Optional[int]:
    if not self.started_at:
      return None
    end_time = self.finished_at or datetime.now(timezone.utc)
    return int((end_time - self.started_at).total_seconds())


@dataclass
class ManagedBotProcess:
  metadata: BotProcessMetadata
  process: asyncio.subprocess.Process
  state: BotProcessState
  stdout_task: asyncio.Task[None]
  stderr_task: asyncio.Task[None]
  monitor_task: asyncio.Task[None]


class BotProcessManager:
  """
  Supervises multiple Passivbot processes per agent.

  Responsibilities:
    - start/stop Passivbot processes via asyncio subprocess APIs
    - tee stdout/stderr to per-bot log files
    - capture exit codes and surface status information
    - ensure proper cleanup on shutdown
  """

  def __init__(self, logs_dir: Path) -> None:
    self._logs_dir = logs_dir
    self._logs_dir.mkdir(parents=True, exist_ok=True)
    self._bots: Dict[str, ManagedBotProcess] = {}
    self._lock = asyncio.Lock()

  async def start_bot(
    self,
    bot_id: str,
    command: Iterable[str],
    *,
    config_path: Path | None = None,
    workdir: Path | None = None,
    env: Dict[str, str] | None = None,
    metadata: Dict[str, Any] | None = None,
  ) -> BotProcessState:
    """
    Start a managed Passivbot process.

    Raises:
      RuntimeError if the bot is already running.
      asyncio.SubprocessError if process launch fails.
    """
    async with self._lock:
      existing = self._bots.get(bot_id)
      if existing and existing.process.returncode is None:
        raise RuntimeError(f"Bot {bot_id} already running")

      command_tuple = tuple(command)
      if not command_tuple:
        raise ValueError("Command must not be empty")

      process_meta = BotProcessMetadata(
        bot_id=bot_id,
        command=command_tuple,
        config_path=config_path,
        workdir=workdir,
        env=dict(env or {}),
      )

      log_path = self._logs_dir / f"{bot_id}.log"
      state = BotProcessState(
        bot_id=bot_id,
        status="starting",
        started_at=datetime.now(timezone.utc),
        log_path=log_path,
      )
      initial_metadata: Dict[str, Any] = {
        "command": list(command_tuple),
        "config_path": str(config_path) if config_path else None,
        "workdir": str(workdir) if workdir else None,
      }
      if metadata:
        initial_metadata.update(metadata)
      if initial_metadata.get("config_path"):
        try:
          initial_metadata.setdefault("config_name", Path(initial_metadata["config_path"]).name)
        except Exception:
          pass
      state.metadata.update(initial_metadata)

      proc_env = os.environ.copy()
      proc_env.update(process_meta.env)

      process = await asyncio.create_subprocess_exec(
        *command_tuple,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(workdir) if workdir else None,
        env=proc_env,
      )
      state.status = "running"

      stdout_task = asyncio.create_task(
        self._pump_stream(process.stdout, log_path, state.log_tail, stream_name="stdout"),
        name=f"{bot_id}-stdout-pump",
      )
      stderr_task = asyncio.create_task(
        self._pump_stream(process.stderr, log_path, state.log_tail, stream_name="stderr"),
        name=f"{bot_id}-stderr-pump",
      )
      monitor_task = asyncio.create_task(
        self._monitor(bot_id, process, state),
        name=f"{bot_id}-monitor",
      )

      managed = ManagedBotProcess(
        metadata=process_meta,
        process=process,
        state=state,
        stdout_task=stdout_task,
        stderr_task=stderr_task,
        monitor_task=monitor_task,
      )
      self._bots[bot_id] = managed
      return state

  async def stop_bot(self, bot_id: str, *, graceful_timeout: float = 15.0) -> BotProcessState:
    """
    Stop a running bot process by sending SIGTERM and waiting for completion.
    """
    async with self._lock:
      managed = self._bots.get(bot_id)
      if not managed:
        raise RuntimeError(f"Bot {bot_id} not found")
      process = managed.process
      if process.returncode is not None:
        return managed.state

      managed.state.status = "stopping"
      process.terminate()

    try:
      await asyncio.wait_for(process.wait(), timeout=graceful_timeout)
    except asyncio.TimeoutError:
      process.kill()
      await process.wait()
      managed.state.last_error = "Process killed after timeout"
    finally:
      await self._finalize_process(bot_id, managed.state, process)
    return managed.state

  async def kill_bot(self, bot_id: str) -> BotProcessState:
    """
    Forcefully stop a bot (SIGKILL). Intended for emergency use.
    """
    async with self._lock:
      managed = self._bots.get(bot_id)
      if not managed:
        raise RuntimeError(f"Bot {bot_id} not found")
      process = managed.process
      if process.returncode is None:
        process.kill()
      await process.wait()
      await self._finalize_process(bot_id, managed.state, process)
      return managed.state

  def is_running(self, bot_id: str) -> bool:
    managed = self._bots.get(bot_id)
    return bool(managed and managed.process.returncode is None)

  def snapshot(self) -> list[Dict[str, Any]]:
    """
    Return serialisable summaries for all managed bots.
    """
    return [managed.state.to_payload() for managed in self._bots.values()]

  def get_state(self, bot_id: str) -> BotProcessState | None:
    managed = self._bots.get(bot_id)
    return managed.state if managed else None

  async def shutdown(self) -> None:
    """
    Stop all managed bots and cancel helper tasks.
    """
    async with self._lock:
      bots = list(self._bots.items())

    for bot_id, managed in bots:
      process = managed.process
      if process.returncode is None:
        process.terminate()
        with contextlib.suppress(asyncio.TimeoutError):
          await asyncio.wait_for(process.wait(), timeout=10)
        if process.returncode is None:
          process.kill()
          await process.wait()
      await self._finalize_process(bot_id, managed.state, process)

  async def update_metadata(self, bot_id: str, updates: Dict[str, Any]) -> BotProcessState:
    async with self._lock:
      managed = self._bots.get(bot_id)
      if not managed:
        raise RuntimeError(f"Bot {bot_id} not found")
      managed.state.metadata.update(updates)
      if updates.get("config_path") and not updates.get("config_name"):
        try:
          managed.state.metadata["config_name"] = Path(str(updates["config_path"])).name
        except Exception:
          pass
      return managed.state

  async def _pump_stream(
    self,
    stream: asyncio.StreamReader | None,
    log_path: Path,
    buffer: Deque[str],
    *,
    stream_name: str,
  ) -> None:
    if stream is None:
      return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
      with log_path.open("a", buffering=1, encoding="utf-8") as log_file:
        while True:
          line = await stream.readline()
          if not line:
            break
          decoded = line.decode("utf-8", errors="replace").rstrip()
          timestamp = datetime.now(timezone.utc).isoformat()
          formatted = f"{timestamp} [{stream_name}] {decoded}"
          try:
            log_file.write(formatted + "\n")
          except Exception:
            pass
          buffer.append(formatted)
    except Exception:
      # Swallow logging exceptions; stream consumption must continue even if file IO fails.
      while True:
        line = await stream.readline()
        if not line:
          break
        decoded = line.decode("utf-8", errors="replace").rstrip()
        timestamp = datetime.now(timezone.utc).isoformat()
        buffer.append(f"{timestamp} [{stream_name}] {decoded}")

  async def _monitor(self, bot_id: str, process: asyncio.subprocess.Process, state: BotProcessState) -> None:
    returncode = await process.wait()
    await self._finalize_process(bot_id, state, process, returncode)

  async def _finalize_process(
    self,
    bot_id: str,
    state: BotProcessState,
    process: asyncio.subprocess.Process,
    returncode: int | None = None,
  ) -> None:
    if returncode is None:
      returncode = process.returncode
    if state.finished_at is not None:
      return

    state.exit_code = returncode
    state.finished_at = datetime.now(timezone.utc)
    state.status = "stopped" if returncode == 0 else "error"
    if returncode not in (0, None):
      state.last_error = f"Process exited with code {returncode}"

    async with self._lock:
      managed = self._bots.get(bot_id)
      if not managed:
        return
      current_task = asyncio.current_task()
      for task in (managed.stdout_task, managed.stderr_task, managed.monitor_task):
        if not task or task.done() or task is current_task:
          continue
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
          await task
