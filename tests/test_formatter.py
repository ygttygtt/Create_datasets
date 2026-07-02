"""Tests for formatter module."""

import pytest
from src.formatter import parse_llm_response, to_llamafactory, to_sharegpt, format_sample


class TestParseLLMResponse:
    def test_direct_json(self):
        raw = '{"instruction": "hello", "input": "", "output": "world"}'
        result = parse_llm_response(raw)
        assert result == {"instruction": "hello", "input": "", "output": "world"}

    def test_markdown_fence(self):
        raw = '```json\n{"instruction": "test", "input": "", "output": "result"}\n```'
        result = parse_llm_response(raw)
        assert result == {"instruction": "test", "input": "", "output": "result"}

    def test_plain_fence(self):
        raw = '```\n{"instruction": "test", "input": "", "output": "result"}\n```'
        result = parse_llm_response(raw)
        assert result == {"instruction": "test", "input": "", "output": "result"}

    def test_empty(self):
        assert parse_llm_response("") is None
        assert parse_llm_response(None) is None

    def test_invalid(self):
        assert parse_llm_response("not json at all") is None


class TestToLLaMAFactory:
    def test_instruction_format(self):
        parsed = {"instruction": "什么是AI?", "input": "", "output": "AI是人工智能..."}
        result = to_llamafactory(parsed, "你是助手")
        assert result["instruction"] == "什么是AI?"
        assert result["output"] == "AI是人工智能..."
        assert result["system"] == "你是助手"
        assert result["history"] == []

    def test_conversation_format(self):
        parsed = {
            "conversations": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
                {"role": "user", "content": "今天天气怎么样"},
                {"role": "assistant", "content": "抱歉，我无法查看实时天气。"},
            ]
        }
        result = to_llamafactory(parsed)
        assert result["instruction"] == "你好"
        assert result["output"] == "抱歉，我无法查看实时天气。"
        assert len(result["history"]) == 1


class TestToShareGPT:
    def test_instruction_format(self):
        parsed = {"instruction": "写一首诗", "input": "", "output": "春眠不觉晓..."}
        result = to_sharegpt(parsed, "你是诗人")
        assert len(result["conversations"]) == 2
        assert result["conversations"][0]["from"] == "human"
        assert result["conversations"][1]["from"] == "gpt"
        assert result["system"] == "你是诗人"

    def test_conversation_format(self):
        parsed = {
            "conversations": [
                {"role": "user", "content": "推荐一本书"},
                {"role": "assistant", "content": "推荐《三体》"},
            ]
        }
        result = to_sharegpt(parsed)
        assert len(result["conversations"]) == 2
        assert result["conversations"][0]["from"] == "human"
        assert result["conversations"][0]["value"] == "推荐一本书"


class TestFormatSample:
    def test_llamafactory_output(self):
        parsed = {"instruction": "test", "input": "", "output": "result"}
        result = format_sample(parsed, "llamafactory")
        assert "instruction" in result
        assert "history" in result

    def test_sharegpt_output(self):
        parsed = {"instruction": "test", "input": "", "output": "result"}
        result = format_sample(parsed, "sharegpt")
        assert "conversations" in result

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            format_sample({}, "unknown_format")
