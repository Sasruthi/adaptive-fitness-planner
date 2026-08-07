"""
FILE LOCATION: backend/app/routers/conversation.py

/api/conversation/start  — POST: initialise conversation, return greeting
/api/conversation/message — POST: send user message, agent decides what to do

CHANGED: now delegates to app.conversation.agent (the unified autonomous
agent) instead of app.conversation.intake_graph (the old slot-filling FSM).
The agent handles open Q&A, profile building, and plan generation itself
via tool-calling — this router no longer needs to know which stage the
conversation is in.
"""
import uuid
from fastapi import APIRouter, HTTPException
from app.schemas.models import (
    StartConversationResponse,
    UserMessageRequest,
    ConversationResponse,
)
from app.conversation.agent import start_conversation, process_user_message

router = APIRouter(prefix="/api/conversation", tags=["Conversation"])


@router.post("/start", response_model=StartConversationResponse)
async def start():
    try:
        thread_id = str(uuid.uuid4())
        result = start_conversation(thread_id)
        return StartConversationResponse(
            thread_id=result["thread_id"],
            stage=result["stage"],
            message=result["message"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start conversation: {str(e)}")


@router.post("/message", response_model=ConversationResponse)
async def message(req: UserMessageRequest):
    """
    Send a user message. The agent autonomously decides whether to answer
    a question via RAG, ask for more profile info, or generate a plan.
    When stage == "plan_ready", `plan` in the response is populated —
    the frontend can render it directly (no separate /api/plan/generate
    call needed; that endpoint remains for programmatic/non-chat use).
    """
    if not req.thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required")
    if not req.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message cannot be empty")

    try:
        result = process_user_message(req.user_message, req.thread_id)
        return ConversationResponse(
            thread_id=result["thread_id"],
            stage=result["stage"],
            message=result["message"],
            slots_complete=result.get("slots_complete", False),
            profile=result.get("profile") or None,
            plan=result.get("plan") or None,
            plan_id=result.get("plan_id"),
            sql_filters=result.get("sql_filters") or None,
            rag_filters=result.get("rag_filters") or None,
            exercises=result.get("exercises") or None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversation error: {str(e)}")
