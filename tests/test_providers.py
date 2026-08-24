import json

from app.models.registry import MODEL_REGISTRY
from app.providers import base, router_client
from app.providers.anthropic_provider import AnthropicAdapter
from app.providers.ollama_provider import OllamaAdapter
from app.providers.openai_provider import OpenAIAdapter


class FakeAdapter:
    def _call(self, prompt, model_config, system=None, max_tokens=None):
        return base.RawResult(text="hi", input_tokens=1000, output_tokens=500)


def test_send_computes_cost_formula():
    model = MODEL_REGISTRY["gpt-5.4-mini"]
    response = base.send(FakeAdapter(), "prompt", model)
    expected_cost = 1000 / 1_000_000 * model.cost_per_1m_input + 500 / 1_000_000 * model.cost_per_1m_output
    assert response.cost_usd == expected_cost
    assert response.input_tokens == 1000
    assert response.output_tokens == 500
    assert response.model_id == model.model_id
    assert response.latency_ms >= 0
    assert response.error is None


def test_send_captures_errors_without_raising():
    class FailingAdapter:
        def _call(self, prompt, model_config, system=None, max_tokens=None):
            raise RuntimeError("boom")

    model = MODEL_REGISTRY["llama3.1:8b"]
    response = base.send(FailingAdapter(), "prompt", model)
    assert response.error == "boom"
    assert response.cost_usd == 0.0


def test_router_client_dispatches_to_correct_adapter(monkeypatch):
    model = MODEL_REGISTRY["claude-sonnet-5"]
    fake = FakeAdapter()
    monkeypatch.setitem(router_client._ADAPTERS, "anthropic", fake)
    response = router_client.send_request("hello", model)
    assert response.text == "hi"
    assert response.provider == "anthropic"


def test_openai_adapter_call(mocker):
    adapter = OpenAIAdapter()
    fake_resp = mocker.Mock()
    fake_resp.choices = [mocker.Mock(message=mocker.Mock(content="answer"))]
    fake_resp.usage = mocker.Mock(prompt_tokens=10, completion_tokens=20)
    fake_client = mocker.Mock()
    fake_client.chat.completions.create.return_value = fake_resp
    adapter._client = fake_client

    result = adapter._call("hi", MODEL_REGISTRY["gpt-5.4-mini"])

    assert result.text == "answer"
    assert result.input_tokens == 10
    assert result.output_tokens == 20
    _, kwargs = fake_client.chat.completions.create.call_args
    assert kwargs["max_completion_tokens"] == base.DEFAULT_MAX_TOKENS


def test_anthropic_adapter_call(mocker):
    adapter = AnthropicAdapter()
    text_block = mocker.Mock(type="text", text="answer")
    fake_resp = mocker.Mock()
    fake_resp.content = [text_block]
    fake_resp.usage = mocker.Mock(input_tokens=15, output_tokens=25)
    fake_client = mocker.Mock()
    fake_client.messages.create.return_value = fake_resp
    adapter._client = fake_client

    result = adapter._call("hi", MODEL_REGISTRY["claude-sonnet-5"])

    assert result.text == "answer"
    assert result.input_tokens == 15
    assert result.output_tokens == 25


def test_ollama_adapter_call(mocker):
    fake_body = json.dumps({"response": "answer", "prompt_eval_count": 5, "eval_count": 7}).encode()
    mock_urlopen = mocker.patch("app.providers.ollama_provider.urllib.request.urlopen")
    mock_urlopen.return_value.__enter__.return_value.read.return_value = fake_body

    result = OllamaAdapter()._call("hi", MODEL_REGISTRY["llama3.1:8b"])

    assert result.text == "answer"
    assert result.input_tokens == 5
    assert result.output_tokens == 7
