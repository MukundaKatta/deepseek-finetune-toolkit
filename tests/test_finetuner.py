"""Tests for FineTuner."""
import pytest
from src.finetuner import FineTuner

def test_init():
    obj = FineTuner()
    stats = obj.get_stats()
    assert stats["total_ops"] == 0

def test_operation():
    obj = FineTuner()
    result = obj.load_model(input="test")
    assert result["processed"] is True
    assert result["operation"] == "load_model"

def test_multiple_ops():
    obj = FineTuner()
    for m in ['load_model', 'prepare_dataset', 'train_lora']:
        getattr(obj, m)(data="test")
    assert obj.get_stats()["total_ops"] == 3

def test_caching():
    obj = FineTuner()
    r1 = obj.load_model(key="same")
    r2 = obj.load_model(key="same")
    assert r2.get("cached") is True

def test_reset():
    obj = FineTuner()
    obj.load_model()
    obj.reset()
    assert obj.get_stats()["total_ops"] == 0

def test_stats():
    obj = FineTuner()
    obj.load_model(x=1)
    obj.prepare_dataset(y=2)
    stats = obj.get_stats()
    assert stats["total_ops"] == 2
    assert "ops_by_type" in stats
