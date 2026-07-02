"""Tests for quality module."""

import pytest
from src.quality import QualityFilter


@pytest.fixture
def default_filter():
    config = {
        "quality": {
            "min_instruction_chars": 10,
            "max_instruction_chars": 512,
            "min_output_chars": 20,
            "max_output_chars": 2048,
            "enable_dedup": True,
            "dedup_method": "exact",
            "fuzzy_threshold": 0.85,
        }
    }
    return QualityFilter(config)


class TestQualityFilter:
    def test_valid_sample(self, default_filter):
        sample = {
            "instruction": "请解释什么是机器学习及其主要应用场景？",
            "output": "机器学习是人工智能的一个重要分支领域，它通过算法让计算机从数据中自动学习和改进。"  # >20 chars
        }
        assert default_filter.is_valid(sample) is True

    def test_instruction_too_short(self, default_filter):
        sample = {"instruction": "?", "output": "这是回答内容，需要至少二十个字才能通过过滤检查。"}
        assert default_filter.is_valid(sample) is False

    def test_output_too_short(self, default_filter):
        sample = {"instruction": "这是一个正常的指令问题", "output": "短"}
        assert default_filter.is_valid(sample) is False

    def test_exact_dedup(self, default_filter):
        assert default_filter.is_duplicate("什么是AI?") is False
        assert default_filter.is_duplicate("什么是AI?") is True  # same text
        assert default_filter.is_duplicate("什么是AI?") is True  # strip() makes it same hash
        assert default_filter.is_duplicate("什么是机器学习") is False  # different text

    def test_fuzzy_dedup(self):
        config = {
            "quality": {
                "min_instruction_chars": 5,
                "max_instruction_chars": 512,
                "min_output_chars": 10,
                "max_output_chars": 2048,
                "enable_dedup": True,
                "dedup_method": "fuzzy",
                "fuzzy_threshold": 0.85,
            }
        }
        qf = QualityFilter(config)
        assert qf.is_duplicate("今天天气怎么样") is False
        assert qf.is_duplicate("今天天气怎么样呢") is True  # very similar

    def test_dedup_disabled(self):
        config = {
            "quality": {
                "min_instruction_chars": 5,
                "max_instruction_chars": 512,
                "min_output_chars": 10,
                "max_output_chars": 2048,
                "enable_dedup": False,
                "dedup_method": "exact",
                "fuzzy_threshold": 0.85,
            }
        }
        qf = QualityFilter(config)
        assert qf.is_duplicate("test") is False
        assert qf.is_duplicate("test") is False  # dedup disabled, always False
