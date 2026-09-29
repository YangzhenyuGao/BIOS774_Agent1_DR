from types import SimpleNamespace

import pytest

from agent1_dr.reporting import extractive_llm


class FakeResponses:
    def __init__(self, output):
        self.output = output

    def create(self, **kwargs):
        assert kwargs["store"] is False
        return SimpleNamespace(output_text=self.output, id="mock")


def test_llm_exact_claim_validation():
    client = SimpleNamespace(responses=FakeResponses('{"sentences":["Measured trust=0.9"]}'))
    assert extractive_llm(["Measured trust=0.9"], "mock", client)["sentences"] == ["Measured trust=0.9"]
    with pytest.raises(ValueError):
        extractive_llm(["Measured trust=0.1"], "mock", client)


def test_llm_credentials_fail_closed(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="no fallback"):
        extractive_llm(["fact"], "mock")
