"""Shared LLM clients with Groq → Azure OpenAI failover on rate limits."""
from __future__ import annotations

import os
from typing import Any, Optional, Sequence

DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def _azure_deployment() -> str:
    return (
        os.getenv("AZURE_OPENAI_DEPLOYMENT")
        or os.getenv("AZURE_DEPLOYMENT")
        or os.getenv("MODEL_NAME")
        or "gpt-4o"
    )


def _azure_api_version() -> str:
    return (
        os.getenv("AZURE_OPENAI_API_VERSION")
        or os.getenv("API_VERSION")
        or "2024-02-01"
    )


def azure_configured() -> bool:
    return bool(os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"))


def is_failover_error(exc: BaseException) -> bool:
    """True for rate-limit / quota / overload errors worth sending to Azure."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "ratelimit" in name or "rate_limit" in name:
        return True
    markers = (
        "429",
        "rate limit",
        "rate_limit",
        "tokens per day",
        "tpm",
        "quota",
        "overloaded",
        "capacity",
        "too many requests",
    )
    return any(m in msg for m in markers)


def get_groq_llm(*, temperature: float = 0.3, max_tokens: int = 600):
    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set — add it to .env")
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_azure_llm(*, temperature: float = 0.3, max_tokens: int = 600):
    from langchain_openai import AzureChatOpenAI

    if not azure_configured():
        raise ValueError(
            "Azure OpenAI is not configured — set AZURE_OPENAI_API_KEY and "
            "AZURE_OPENAI_ENDPOINT (plus AZURE_OPENAI_DEPLOYMENT or AZURE_DEPLOYMENT)"
        )
    return AzureChatOpenAI(
        azure_deployment=_azure_deployment(),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=_azure_api_version(),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_ollama_llm(*, temperature: float = 0.3, max_tokens: int = 600):
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        temperature=temperature,
    )


def resolve_provider(provider: Optional[str] = None) -> str:
    return (provider or os.getenv("LLM_PROVIDER", "groq")).lower()


def get_llm(
    *,
    temperature: float = 0.3,
    max_tokens: int = 600,
    provider: Optional[str] = None,
    with_azure_fallback: bool = True,
):
    """
    Return a chat model for the requested provider.

    When provider is groq (default) and Azure is configured, wraps Groq with
    LangChain `with_fallbacks` so rate-limits/quota errors automatically retry
    on Azure OpenAI.
    """
    p = resolve_provider(provider)

    if p in ("azure", "openai"):
        return get_azure_llm(temperature=temperature, max_tokens=max_tokens)
    if p == "ollama":
        return get_ollama_llm(temperature=temperature, max_tokens=max_tokens)

    # Default: Groq
    primary = get_groq_llm(temperature=temperature, max_tokens=max_tokens)
    if with_azure_fallback and azure_configured():
        fallback = get_azure_llm(temperature=temperature, max_tokens=max_tokens)
        print("[LLM] Groq primary, Azure OpenAI configured as failover")
        return primary.with_fallbacks([fallback], exceptions_to_handle=(Exception,))
    return primary

def get_llm_with_tools(
    tools: Sequence[Any],
    *,
    temperature: float = 0.3,
    max_tokens: int = 4000,
    provider: Optional[str] = None,
    with_azure_fallback: bool = True,
):
    """Bind tools on primary (and Azure fallback when enabled)."""
    p = resolve_provider(provider)

    if p in ("azure", "openai"):
        return get_azure_llm(temperature=temperature, max_tokens=max_tokens).bind_tools(tools)
    if p == "ollama":
        return get_ollama_llm(temperature=temperature, max_tokens=max_tokens).bind_tools(tools)

    primary = get_groq_llm(temperature=temperature, max_tokens=max_tokens).bind_tools(tools)
    if with_azure_fallback and azure_configured():
        fallback = get_azure_llm(temperature=temperature, max_tokens=max_tokens).bind_tools(tools)
        print("[LLM] Groq+tools primary, Azure OpenAI+tools as failover")
        return primary.with_fallbacks([fallback], exceptions_to_handle=(Exception,))
    return primary


def invoke_with_failover(primary_invoke, azure_invoke, *args, **kwargs):
    """
    Call primary_invoke; on rate-limit/quota errors, call azure_invoke once.
    Useful when the LLM is baked into a cached agent graph.
    """
    try:
        return primary_invoke(*args, **kwargs)
    except Exception as e:
        if not (is_failover_error(e) and azure_configured()):
            raise
        print(f"[LLM] Primary failed ({type(e).__name__}: {e}); retrying with Azure OpenAI")
        return azure_invoke(*args, **kwargs)
