"""Tests for DeepseekFinetuneToolkit."""
from src.core import DeepseekFinetuneToolkit
def test_init(): assert DeepseekFinetuneToolkit().get_stats()["ops"] == 0
def test_op(): c = DeepseekFinetuneToolkit(); c.process(x=1); assert c.get_stats()["ops"] == 1
def test_multi(): c = DeepseekFinetuneToolkit(); [c.process() for _ in range(5)]; assert c.get_stats()["ops"] == 5
def test_reset(): c = DeepseekFinetuneToolkit(); c.process(); c.reset(); assert c.get_stats()["ops"] == 0
def test_service_name(): c = DeepseekFinetuneToolkit(); r = c.process(); assert r["service"] == "deepseek-finetune-toolkit"
