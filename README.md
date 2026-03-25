# Token Benchmark

AI 模型 Token 生成速度基准测试工具。支持 OpenAI / Anthropic / DeepSeek / Kimi 及任何 OpenAI 兼容接口。

## 功能特性

- **多模型支持**：OpenAI GPT 系列、Anthropic Claude 系列、DeepSeek、Moonshot Kimi
- **自定义端点**：支持任何 OpenAI 兼容 API（Ollama、vLLM、自建模型等）
- **实时可视化**：Streaming 过程中实时显示 Token 计数和吞吐量
- **精确计量**：使用 `tiktoken` 计算真实 Token 数（非 chunk 数量）
- **多轮取中位数**：排除网络波动干扰，默认 3 轮测试
- **终端表格**：美观的 Rich 表格输出
- **HTML 报告**：可选生成带 Chart.js 对比图表的 Web 报告

## 安装

```bash
git clone https://github.com/G3niusYukki/token-benchmark.git
cd token-benchmark
pip install -r requirements.txt
```

需要 Python 3.10+

## 快速开始

```bash
python main.py -p openai
```

交互式输入 API Key 和模型名称即可开始测试。

## 使用方法

### 基础用法

```bash
# 测试所有已配置的 provider
python main.py

# 只测试 OpenAI
python main.py -p openai

# 同时测试多个 provider
python main.py -p openai anthropic deepseek

# 指定测试轮次
python main.py -p openai -r 5
```

### 生成 HTML 报告

```bash
python main.py -p openai anthropic --html -o report.html
```

### 实时Verbose 模式

```bash
python main.py -p openai -r 1 -v
```

Streaming 过程中实时显示 Token 计数和吞吐量，测试结束后显示响应预览和计算明细。

### 自定义 API Endpoint

```bash
python main.py -p openai

# 交互式配置
# API Key: sk-xxx
# Model: my-model
# Base URL: https://api.example.com/v1
```

支持任何 OpenAI 兼容接口，例如：
- [Ollama](https://ollama.com/)：`http://localhost:11434/v1`
- [vLLM](https://docs.vllm.ai/)：`http://localhost:8000/v1`
- [SiliconFlow](https://siliconflow.cn/)：`https://api.siliconflow.cn/v1`

### 配置文件

也可以预先配置 `config.yaml`：

```yaml
providers:
  openai:
    api_key: "sk-xxx"
    model: "gpt-4o"
    base_url: "https://api.openai.com/v1"

benchmark:
  prompt: "请用100字介绍一下人工智能的发展历史。"
  rounds: 3
  timeout: 60
```

## 输出指标说明

| 指标 | 说明 |
|------|------|
| **TTFT (ms)** | Time to First Token，首 Token 延迟，模型开始响应的时间 |
| **Tokens/s** | 每秒生成 Token 数，核心吞吐量指标 |
| **Total (ms)** | 总响应时间，从发请求到最后一个 Token 的时间 |
| **Total Tokens** | 响应的总 Token 数量 |
| **Status** | 测试成功/失败 |

### Token 计数原理

Streaming 过程中实时拼接所有文本片段，使用 `tiktoken` (cl100k_base) 编码器计算真实 Token 数。

```
Tokens/s = 总 Token 数 / Streaming 总耗时
```

## 项目结构

```
token-benchmark/
├── benchmark/
│   ├── models.py          # BenchmarkResult 数据类
│   ├── runner.py          # 测试运行器
│   ├── reporter.py        # 终端 + HTML 报告
│   └── providers/         # 各模型适配器
│       ├── openai.py
│       ├── anthropic.py
│       ├── deepseek.py
│       └── kimi.py
├── templates/
│   └── report.html        # HTML 报告模板
├── config.yaml           # 配置文件
├── main.py               # CLI 入口
└── requirements.txt
```

## 添加新的 Provider

实现 `BaseProvider` 抽象类即可：

```python
from benchmark.providers.base import BaseProvider
from benchmark.models import BenchmarkResult

class MyProvider(BaseProvider):
    name = "myprovider"

    def run(self, prompt: str, timeout: int = 60) -> BenchmarkResult:
        # 1. 记录开始时间 t0
        # 2. 发请求，Streaming 接收
        # 3. 拼接 full_text
        # 4. 用 tiktoken 计算 token 数
        # 5. 返回 BenchmarkResult
```

然后在 `benchmark/providers/__init__.py` 中注册。

## License

MIT
