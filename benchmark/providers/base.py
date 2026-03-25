import time
from abc import ABC, abstractmethod
from benchmark.models import BenchmarkResult

class BaseProvider(ABC):
    name: str = "base"

    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model
        self.extra = kwargs

    @abstractmethod
    def run(self, prompt: str, timeout: int = 60) -> BenchmarkResult:
        """执行流式请求并返回基准测试结果"""
        pass

    def _calc_tps(self, token_count: int, elapsed: float) -> float:
        return token_count / elapsed if elapsed > 0 else 0.0
