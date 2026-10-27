import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from knowflow.app.api import router
from knowflow.app.batch_api import router as batch_router
from knowflow.app.config import get_settings
from knowflow.app.conversation_api import router as conversation_router
from knowflow.app.conversation_review_api import router as conversation_review_router
from knowflow.app.db import engine
from knowflow.app.diagnostics_api import router as diagnostics_router
from knowflow.app.ingestion_jobs import worker_loop
from knowflow.app.memory_api import router as memory_router
from knowflow.app.task_alignment_api import router as task_alignment_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    workers = [asyncio.create_task(worker_loop()) for _ in range(get_settings().ingestion_workers)]
    try:
        yield
    finally:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        await engine.dispose()


app = FastAPI(
    title=get_settings().app_name,
    version="0.1.0",
    description="Evidence-grounded document retrieval and question answering.",
    lifespan=lifespan,
)
app.include_router(router)
app.include_router(batch_router)
app.include_router(diagnostics_router)
app.include_router(conversation_router)
app.include_router(conversation_review_router)
app.include_router(memory_router)
app.include_router(task_alignment_router)


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Database is unavailable") from exc
    return {"status": "ok"}
