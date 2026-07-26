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

    # Pre-check: detect greetings and off-topic before calling AI
    question_lower = request.question.lower().strip()
    greetings = {"hi", "hello", "hey", "hiya", "howdy", "sup", "yo"}
    is_greeting = question_lower in greetings or any(question_lower.startswith(g + " ") for g in greetings)
    if is_greeting:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        greeting_msg = ChatMessage(
            dataset_id=request.dataset_id,
            user_id=current_user_id,
            role="assistant",
            content="Hi there! I am Querify, your AI data analyst. Ask me anything about your dataset - try: Show top 5 rows, Count by category, or Average value by group.",
            result_data=None,
            operation_spec=None,
            execution_time_ms=0,
        )
        user_msg_g = ChatMessage(
            dataset_id=request.dataset_id,
            user_id=current_user_id,
            role="user",
            content=request.question,
        )
        db.add(user_msg_g)
        db.add(greeting_msg)
        await db.flush()
        await db.commit()
        return ChatQueryResponse(
            message_id=greeting_msg.id,
            question=request.question,
            answer=greeting_msg.content,
            result_data=None,
            execution_time_ms=0,
            created_at=now,
        )

    from app.services.usage_service import check_and_increment
    await check_and_increment(db, current_user_id, 'query')
    service = NLQueryService()
    try:
        query_result = await service.execute_query(df, request.question, str(dataset.id))
    except Exception as e:
        err = str(e).lower()
        # Detect greetings vs genuinely off-topic queries
        question_lower = request.question.lower().strip()
        greetings = {"hi", "hello", "hey", "hiya", "howdy", "sup", "what's up", "whats up", "good morning", "good afternoon", "good evening"}
        is_greeting = question_lower in greetings or any(question_lower.startswith(g) for g in greetings)
        if is_greeting:
            friendly = (
                f"Hi there! I am Querify, your AI data analyst. "
                f"I can help you explore your dataset - try asking me things like: "
                f"Show top 5 rows by value, Count by category, or Average sales by region."
            )
        else:
            friendly = (
                "I can only answer questions about your dataset. "
                "Try something like: Show top 5 by a column, Count by category, or Average value by group."
            )
        now = datetime.now(timezone.utc)
        fallback_msg = ChatMessage(
            dataset_id=request.dataset_id,
            user_id=current_user_id,
            role="assistant",
            content=friendly,
            result_data=None,
            operation_spec=None,
            execution_time_ms=0,
        )
        user_msg = ChatMessage(
            dataset_id=request.dataset_id,
            user_id=current_user_id,
            role="user",
            content=request.question,
        )
        db.add(user_msg)
        db.add(fallback_msg)
        await db.flush()
        await db.commit()
        return ChatQueryResponse(
            message_id=fallback_msg.id,
            question=request.question,
            answer=friendly,
            result_data=None,
            execution_time_ms=0,
            created_at=now,
        )

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


@router.get("/suggestions/{dataset_id}")
async def get_suggestions(
    dataset_id: uuid.UUID,
    dataset_and_df=Depends(load_dataset_df),
    current_user_id: uuid.UUID = Depends(get_current_user),
):
    from app.services.ai.groq_client import GroqClient
    dataset, df = dataset_and_df
    cols = []
    for col in df.columns[:8]:
        sample = df[col].dropna().head(3).tolist()
        cols.append({"name": col, "dtype": str(df[col].dtype), "sample": sample})
    prompt = f"""You are a data analyst. Given this dataset schema, suggest exactly 3 short, specific natural language questions a user might ask.

Dataset: {dataset.name}
Columns: {cols}

Return ONLY a JSON array of 3 strings. No explanation. No markdown. Example:
["Top 5 companies by revenue", "Average salary by department", "Show sales trend over time"]

JSON array only:"""
    client = GroqClient()
    import json, re
    text = await client.get_insights(prompt)
    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    try:
        suggestions = json.loads(cleaned)
        if isinstance(suggestions, list):
            return {"suggestions": suggestions[:3]}
    except Exception:
        pass
    return {"suggestions": ["Show top values", "Count by category", "Show trends over time"]}


@router.post("/chart-insight")
async def chart_insight(
    request: dict,
    current_user_id: uuid.UUID = Depends(get_current_user),
):
    from app.services.ai.groq_client import GroqClient
    title = request.get("title", "Chart")
    summary = request.get("summary", "")
    prompt = f"""You are a business data analyst. Analyze this chart data and give a concise 3-4 sentence business insight.

Chart: "{title}"
Data summary: {summary}

Be specific, mention actual values, identify patterns or outliers. Write clear prose, no bullet points."""
    client = GroqClient()
    text = await client.get_insights(prompt)
    return {"insight": text}
