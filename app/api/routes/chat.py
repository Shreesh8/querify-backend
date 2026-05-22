"""api/routes/chat.py"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import get_current_user
from app.api.dependencies.dataset import load_dataset_df
from app.db.models.dataset import ChatMessage
from app.db.session import get_db
from app.schemas.dataset import ChatQueryRequest, ChatQueryResponse
from app.services.ai.query_service import NLQueryService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "/query",
    response_model=ChatQueryResponse,
    summary="Ask a natural language question about a dataset",
)
async def query_dataset(
    request: ChatQueryRequest,
    db: AsyncSession = Depends(get_db),
    current_user_id: uuid.UUID = Depends(get_current_user),
):
    # Load dataset + df via shared dependency
    from app.api.dependencies.dataset import get_dataset_or_404, _read_file
    from pathlib import Path
    import asyncio

    from sqlalchemy import select
    from app.db.models.dataset import Dataset

    result = await db.execute(select(Dataset).where(Dataset.id == request.dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        from fastapi import HTTPException, status
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.")

    file_path = Path(dataset.file_path)
    df = await asyncio.get_event_loop().run_in_executor(
        None, _read_file, file_path, dataset.file_type
    )

    service = NLQueryService()
    query_result = await service.execute_query(df, request.question, str(dataset.id))

    # Persist conversation history
    user_msg = ChatMessage(
        dataset_id=request.dataset_id,
        user_id=current_user_id,
        role="user",
        content=request.question,
    )
    assistant_msg = ChatMessage(
        dataset_id=request.dataset_id,
        user_id=current_user_id,
        role="assistant",
        content=query_result["answer"],
        result_data=query_result["result_data"],
        operation_spec=query_result["operation_spec"],
        execution_time_ms=query_result["execution_time_ms"],
    )
    db.add(user_msg)
    db.add(assistant_msg)
    await db.flush()

    return ChatQueryResponse(
        message_id=assistant_msg.id,
        question=request.question,
        answer=query_result["answer"],
        result_data=query_result["result_data"],
        execution_time_ms=query_result["execution_time_ms"],
        created_at=assistant_msg.created_at,
    )
