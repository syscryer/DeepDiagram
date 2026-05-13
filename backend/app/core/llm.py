from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from app.core.config import settings


def _is_anthropic_url(url: str) -> bool:
    """根据 URL 判断是否为 Anthropic 格式接口"""
    if not url:
        return False
    url_lower = url.lower()
    anthropic_keywords = ["claude", "anthropic", "/claude/"]
    return any(kw in url_lower for kw in anthropic_keywords)


def _create_llm(api_key: str, base_url: str | None, model: str, temperature: float, max_tokens: int):
    """根据 base_url 自动选择 OpenAI 或 Anthropic 格式"""
    if base_url and _is_anthropic_url(base_url):
        from app.core.logger import logger
        logger.info(f"Using Anthropic provider for base_url: {base_url}")
        return ChatAnthropic(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            streaming=True,
            timeout=120,
            max_tokens=max_tokens
        )

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        streaming=True,
        request_timeout=120,
        max_tokens=max_tokens
    )


def get_llm(model_name: str | None = None, temperature: float = 0.3, api_key: str | None = None, base_url: str | None = None):
    """
    根据配置返回 LLM 实例，自动识别 OpenAI / Anthropic / DeepSeek 格式
    """

    max_tokens = settings.MAX_TOKENS

    final_api_key = api_key.strip() if (api_key and api_key.strip()) else None
    if final_api_key and final_api_key.startswith("Bearer "):
        final_api_key = final_api_key[7:].strip()

    final_base_url = base_url.strip() if (base_url and base_url.strip()) else None
    if final_base_url:
        final_base_url = final_base_url.rstrip("/")
        for suffix in ["/chat/completions", "/completions"]:
            if final_base_url.lower().endswith(suffix):
                final_base_url = final_base_url[:-len(suffix)].rstrip("/")
                break

    final_model = (model_name.strip() if model_name else None) or settings.MODEL_ID

    if final_api_key:
        return _create_llm(final_api_key, final_base_url, final_model or "claude-sonnet-4-20250514", temperature, max_tokens)

    # Priority: DeepSeek if key is present
    if settings.DEEPSEEK_API_KEY:
        model = settings.MODEL_ID or "deepseek-chat"
        return _create_llm(settings.DEEPSEEK_API_KEY, settings.DEEPSEEK_BASE_URL, model, temperature, max_tokens)

    # Fallback to OpenAI
    return _create_llm(
        settings.OPENAI_API_KEY,
        settings.OPENAI_BASE_URL,
        model_name or settings.MODEL_ID or "claude-sonnet-4-20250514",
        temperature,
        max_tokens
    )


def get_configured_llm(state: "AgentState", temperature: float = 0.3):
    """
    Helper to get an LLM instance based on the current AgentState configuration.
    """
    config = state.get("model_config")
    from app.core.logger import logger
    logger.info(f"DEBUG | get_configured_llm | config found: {True if config else False}")
    if config:
        return get_llm(
            model_name=config.get("model_id"),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            temperature=temperature
        )
    return get_llm(temperature=temperature)


def get_time_instructions() -> str:
    """
    Returns the current time context.
    """
    from datetime import datetime, timezone
    import calendar
    
    now = datetime.now(timezone.utc)
    day_name = calendar.day_name[now.weekday()]
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S UTC")
    
    return f"\n\n### CURRENT TIME CONTEXT\n- Current Date and Time: {formatted_time}\n- Day of Week: {day_name}"

def get_thinking_instructions() -> str:
    """
    Returns system prompt instructions based on thinking verbosity setting,
    plus the current time context.
    """
    verbosity = settings.THINKING_VERBOSITY.lower()
    
    time_context = get_time_instructions()
    thinking_part = ""
    
    if verbosity == "concise":
        thinking_part = "\n\n### THINKING PROCESS\n- Please be extremely concise in your internal thinking (<think> tags).\n- Focus ONLY on critical reasoning steps.\n- Avoid restating the obvious or verbose planning."
    elif verbosity == "verbose":
        thinking_part = "\n\n### THINKING PROCESS\n- Please explore all possibilities in your internal thinking.\n- Verify assumptions and plan in detail."
    
    return thinking_part + time_context
