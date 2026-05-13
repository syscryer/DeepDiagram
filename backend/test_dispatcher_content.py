import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from app.agents.dispatcher import router_node


class DummyLlm:
    async def ainvoke(self, messages):
        return SimpleNamespace(
            content=[
                "",
                {"type": "thinking", "thinking": "The user is asking for a flowchart."},
                "flow",
            ]
        )


class TestDispatcherContent(unittest.IsolatedAsyncioTestCase):
    async def test_router_handles_anthropic_list_content(self):
        state = {
            "messages": [HumanMessage(content="帮我画一个登录流程图")],
            "model_config": {
                "model_id": "deepseek-v4-pro",
                "api_key": "sk-test",
                "base_url": "https://api.deepseek.com/anthropic",
            },
        }

        with patch("app.agents.dispatcher.get_configured_llm", return_value=DummyLlm()):
            result = await router_node(state)

        self.assertEqual(result, {"intent": "flowchart"})


if __name__ == "__main__":
    unittest.main()
