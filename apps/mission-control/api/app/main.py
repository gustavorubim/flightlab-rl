"""FastAPI entrypoint for the mission-control backend."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.config import controller_registry_path
from app.schemas import (
    CommandResponseModel,
    ControllerOptionModel,
    MissionModel,
    SessionSnapshotModel,
    SessionStartRequest,
)
from app.session import MissionRuntimeService


def create_app() -> FastAPI:
    """Create the FastAPI application."""

    runtime = MissionRuntimeService(controller_registry_path=controller_registry_path())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.mission_runtime = runtime
        await runtime.startup()
        try:
            yield
        finally:
            await runtime.shutdown()

    app = FastAPI(title="FlightLab Mission Control", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/session", response_model=SessionSnapshotModel)
    async def get_session() -> SessionSnapshotModel:
        return await app.state.mission_runtime.get_snapshot()

    @app.get("/api/controllers", response_model=list[ControllerOptionModel])
    async def get_controllers() -> list[ControllerOptionModel]:
        return await app.state.mission_runtime.list_controllers()

    @app.post("/api/session/start", response_model=CommandResponseModel)
    async def start_session(request: SessionStartRequest) -> CommandResponseModel:
        return await app.state.mission_runtime.start_session(request)

    @app.post("/api/session/reset", response_model=CommandResponseModel)
    async def reset_session() -> CommandResponseModel:
        return await app.state.mission_runtime.reset_session()

    @app.post("/api/commands/takeoff", response_model=CommandResponseModel)
    async def takeoff() -> CommandResponseModel:
        return await app.state.mission_runtime.arm_takeoff()

    @app.post("/api/commands/pause", response_model=CommandResponseModel)
    async def pause() -> CommandResponseModel:
        return await app.state.mission_runtime.pause()

    @app.post("/api/commands/resume", response_model=CommandResponseModel)
    async def resume() -> CommandResponseModel:
        return await app.state.mission_runtime.resume()

    @app.put("/api/mission", response_model=CommandResponseModel)
    async def update_mission(mission: MissionModel) -> CommandResponseModel:
        return await app.state.mission_runtime.replace_mission(mission)

    @app.websocket("/ws/telemetry")
    async def telemetry(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                snapshot = await app.state.mission_runtime.get_snapshot()
                await websocket.send_json(snapshot.model_dump(mode="json"))
                await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            return

    return app


app = create_app()
