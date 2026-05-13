import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.api.routes import TestModelRequest, test_model_connection


class DummyLlm:
    async def ainvoke(self, messages):
        return SimpleNamespace(content="OK")


class FailingLlm:
    async def ainvoke(self, messages):
        raise Exception("Error code: 404 - {'error': {'message': 'Unknown model deepseek-v4-pro'}}")


class TestModelConnection(unittest.IsolatedAsyncioTestCase):
    async def test_uses_shared_llm_factory_for_custom_provider_urls(self):
        request = TestModelRequest(
            model_id="deepseek-v4-pro",
            api_key="sk-test",
            base_url="https://api.deepseek.com/anthropic",
        )

        with patch("app.api.routes.get_llm", return_value=DummyLlm(), create=True) as get_llm:
            result = await test_model_connection(request)

        self.assertTrue(result["success"])
        get_llm.assert_called_once_with(
            model_name="deepseek-v4-pro",
            api_key="sk-test",
            base_url="https://api.deepseek.com/anthropic",
            temperature=0,
        )

    async def test_preserves_provider_error_detail_for_not_found_responses(self):
        request = TestModelRequest(
            model_id="deepseek-v4-pro",
            api_key="sk-test",
            base_url="https://api.deepseek.com/anthropic",
        )

        with patch("app.api.routes.get_llm", return_value=FailingLlm(), create=True):
            result = await test_model_connection(request)

        self.assertFalse(result["success"])
        self.assertIn("Provider returned 404", result["message"])
        self.assertIn("Unknown model deepseek-v4-pro", result["message"])


if __name__ == "__main__":
    unittest.main()
