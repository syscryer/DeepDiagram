from typing import Any


def content_to_text(content: Any) -> str:
    """Convert LangChain message content into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item:
                    parts.append(item)
                continue
            if isinstance(item, dict):
                if item.get("type") == "thinking":
                    continue
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
                continue
            if item is not None:
                parts.append(str(item))
        return "".join(parts)
    return str(content)
