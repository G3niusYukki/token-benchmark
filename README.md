# Token Benchmark

AI 模型 Token 生成速度基准测试工具。支持 **OpenAI**、**Anthropic**、**DeepSeek**、**Kimi** 以及任何 **OpenAI 兼容**或 **Anthropic 兼容** 的 API 端点(Ollama、vLLM、OpenRouter、自建代理…)。输入 Key + URL,工具会自动识别端点风格、拉取可用模型,让你用交互式菜单多选后开始测试。

## 功能特性

- **🔍 端点自动检测**: 给一个 URL + Key,自动嗅探是 OpenAI Chat Completions 风格还是 Anthropic Messages 风格,匹配不上时退回 host-name 启发式
- **📋 模型自动拉取**: 通过 SDK 拉取 `/v1/models`,过滤掉 embedding / 图像 / 音频 / TTS 模型,只保留可聊天的模型
- **✅ 交互式多选**: 真正的 prompt_toolkit TUI 菜单 —— 空格切换, `a` 全选, `i` 反选, `/` 模糊过滤, Enter 确认
- **🔁 多端点串联**: 在一次会话里添加多个端点(OpenAI + Anthropic + 本地 Ollama)然后一起跑
- **⏱ 精确计量**: 使用 `tiktoken cl100k_base` 计算真实 token 数,不是 chunk 数
- **📊 实时可视化**: 终端 Rich Live 面板显示 TTFT、累计 token、实时 TPS
- **📈 多轮取中位数**: 3 轮默认,排除网络抖动
- **📑 HTML 报告**: 可选生成带 Chart.js 折线/柱状对比的 Web 报告
- **🧩 统一 Provider 层**: 一个 `UniversalProvider` 类同时驱动 OpenAI 和 Anthropic SDK,无重复代码

## 安装

```bash
git clone https://github.com/G3niusYukki/token-benchmark.git
cd token-benchmark
pip install -r requirements.txt
```

需要 Python 3.10+。

## 快速开始

### 方式一:交互式 on-boarding (推荐)

直接运行,无任何参数,工具会一步步引导你:

```bash
python main.py
```

流程:

1. 选择起点:OpenAI / Anthropic / DeepSeek / Kimi / 自定义 URL
2. 输入 Base URL 和 API Key(API Key 隐藏输入)
3. **自动检测端点风格** 并显示结果
4. **拉取可用模型列表** 展示给你
5. **多选要测试的模型**(默认全部选中, 空格切换, a 全选, / 过滤, Enter 确认)
6. 可选添加下一个端点
7. 确认后开始跑,跑完打印表格 + (可选) HTML 报告

非 TTY 环境下(CI、脚本)会自动降级为 `input()` 编号选择。

### 方式二:命令行一行配置 (CI 友好)

跳过交互,直接用 `--add-endpoint` 预置端点:

```bash
python main.py \
  --add-endpoint "owner=openai|key=$OPENAI_KEY|url=https://api.openai.com/v1|style=openai|model=gpt-4o,gpt-4o-mini" \
  --add-endpoint "owner=anthropic|key=$ANTHROPIC_KEY|url=https://api.anthropic.com|style=anthropic|model=claude-sonnet-4-5-latest" \
  -r 3 -v --html -o report.html
```

可重复 `--add-endpoint` 添加多个端点。

### 方式三:传统 `config.yaml` (向后兼容)

```bash
python main.py -p openai anthropic
```

`config.yaml` 写法保持不变:

```yaml
providers:
  openai:
    api_key: "sk-xxx"
    model: "gpt-4o"
    base_url: "https://api.openai.com/v1"
  anthropic:
    api_key: "sk-ant-xxx"
    model: "claude-sonnet-4-5-latest"

benchmark:
  prompt: "请用100字介绍一下人工智能的发展历史。"
  rounds: 3
  timeout: 60
```

## CLI 完整参数

| 参数 | 说明 |
|------|------|
| `-p PROVIDERS [PROVIDERS ...]` | 走 config.yaml 旧路径,只跑指定 provider |
| `-r ROUNDS` | 每模型跑几轮,默认 3 |
| `-c CONFIG` | 旧模式下的配置文件路径,默认 `config.yaml` |
| `--html` | 生成 HTML 报告 |
| `-o OUTPUT` | HTML 报告输出路径,默认 `benchmark_report.html` |
| `-v` / `--verbose` | 实时显示 streaming token + 计算明细 |
| `--prompt TEXT` | 覆盖 prompt,默认从 config 读 |
| `--timeout SEC` | 单次请求超时,默认 60 |
| `--add-endpoint SPEC` | 预设端点,可重复;SPEC 格式 `owner=k\|key=k\|url=u\|style=s\|model=m1,m2` |
| `--no-interactive` | 强制用 `input()` 而非 TUI |
| `--list-models` | 旧模式下只打印已配置的 model, 不跑测试 |

## 支持的端点

工具会探测以下常见端点;不在这列表里的也能用,只要它返回 OpenAI 或 Anthropic 风格的 `/v1/models`:

| 服务 | URL | 风格 | 默认端口 |
|------|-----|------|----------|
| OpenAI | `https://api.openai.com/v1` | openai | - |
| Anthropic | `https://api.anthropic.com` | anthropic | - |
| DeepSeek | `https://api.deepseek.com/v1` | openai | - |
| Kimi / Moonshot | `https://api.moonshot.cn/v1` | openai | - |
| SiliconFlow | `https://api.siliconflow.cn/v1` | openai | - |
| Groq | `https://api.groq.com/openai/v1` | openai | - |
| Together | `https://api.together.xyz/v1` | openai | - |
| Fireworks | `https://api.fireworks.ai/inference/v1` | openai | - |
| OpenRouter | `https://openrouter.ai/api/v1` | openai | - |
| Ollama | `http://localhost:11434/v1` | openai | 11434 |
| vLLM | `http://localhost:8000/v1` | openai | 8000 |
| LM Studio | `http://localhost:1234/v1` | openai | 1234 |
| Google Gemini (compat) | `https://generativelanguage.googleapis.com/v1beta/openai` | openai | - |

## 输出指标

| 指标 | 说明 |
|------|------|
| **TTFT (ms)** | Time to First Token,首 token 延迟 |
| **Tokens/s** | 每秒生成 token,核心吞吐量指标 |
| **Total (ms)** | 总响应时间 |
| **Total Tokens** | 响应的总 token 数量(tiktoken 真实计数) |
| **Status** | ✅ 成功 / ❌ 失败 |

```
Tokens/s = Total Tokens / (t_last_token - t0)
```

## 项目结构

```
token-benchmark/
├── benchmark/
│   ├── endpoint_detector.py     # 自动嗅探 OpenAI vs Anthropic
│   ├── model_fetcher.py         # 拉取 /v1/models 并归一化
│   ├── menu.py                  # prompt_toolkit 交互式菜单
│   ├── onboarding.py            # 端到端 on-boarding 流程
│   ├── models.py                # BenchmarkResult 数据类
│   ├── runner.py                # 测试调度 (run_from_endpoints / run_from_config)
│   ├── reporter.py              # 终端表格 + HTML 报告
│   └── providers/
│       ├── universal.py         # 双风格统一 Provider
│       ├── openai.py            # → UniversalProvider 委托
│       ├── anthropic.py         # → UniversalProvider 委托
│       ├── deepseek.py          # → UniversalProvider 委托
│       ├── kimi.py              # → UniversalProvider 委托
│       └── base.py              # 抽象基类
├── templates/
│   └── report.html              # HTML 报告模板
├── tests/                       # pytest 单元测试 (33 tests)
├── main.py                      # CLI 入口
├── config.yaml                  # 可选的传统配置
└── requirements.txt
```

## 添加新的 Provider

不要再写新类 —— 直接用 `UniversalProvider`:

```python
from benchmark.endpoint_detector import EndpointStyle
from benchmark.providers.universal import UniversalProvider

p = UniversalProvider(
    api_key="sk-xxx",
    model="my-model",
    base_url="https://api.example.com/v1",
    style=EndpointStyle.OPENAI,    # or EndpointStyle.ANTHROPIC
    owner="myprovider",
    verbose=True,
)
result = p.run("Hello!", timeout=30)
```

## 错误处理

- 网络超时 → 异常被 Provider 捕获,返回 `success=False, error="Timeout: ..."`
- 401 / 403 → 通常是 Key 错; 端点探测阶段会先告诉你 Key 是否有问题
- `/v1/models` 返回 404 → 自动使用预置的模型目录(fallback catalog)
- HTTP 不可达 → 返回 `EndpointStyle.UNKNOWN` 加上 note 字段解释

## 测试

```bash
python -m pytest tests/ -v
```

33 个测试覆盖:URL 归一化、主机启发式、响应体形状判别、探测流程(用 mock)、OpenAI/Anthropic 模型拉取、聊天模型过滤器、UniversalProvider streaming + 异常路径、BenchmarkResult 摘要。

## License

MIT

---

## 新增: 生产级指标 (v2)

v2 在原版基础上加了 Langfuse / OpenLLMetry 风格的生产级指标:

| 指标 | 公式 | 用途 |
|------|------|------|
| **TTFT (ms)** | 首 chunk 到达时间 | 反映冷启动 + 网络延迟 |
| **Tokens/s** | total_tokens / (t_last - t0) | 聚合吞吐量 |
| **P50/P95/P99 ITL (ms)** | Inter-Token Latency 的分位数 | 衡量流是否"丝滑" |
| **TPOT (ms)** | (total - ttft) / (tokens - 1) | 每个 token 平均生成时间 |
| **Cold start (ms)** | 首轮总耗时 | 反映 JIT / 连接建立 |
| **Steady TPS** | 第 2 轮后中位 TPS | 反映热路径性能 |
| **Prompt tokens** | API 返回的 usage | 算成本 |
| **Completion tokens** | API 返回的 usage | 算成本 |
| **Cost USD** | `prompt × input_price + completion × output_price` | 真实成本对比 |

### 新 CLI 选项

```bash
python main.py --warmup 2 -r 5 --export results.json   # 2 轮预热 + 5 轮测量 + JSON 导出
python main.py --export results.csv                    # CSV 给 pandas / Excel
```

`--warmup` 默认 1 轮丢弃(避免 JIT / TLS handshake 污染测量)。

### 成本定价覆盖

```bash
export TOKEN_BENCHMARK_PRICING_OVERRIDE='{"gpt-4o":{"input":2.5,"output":10.0}}'
```

### JSON 导出 schema (v1)

```json
{
  "schema": "token-benchmark/v1",
  "results": [
    {
      "provider": "openai", "model": "gpt-4o",
      "ttft_ms": 234, "tokens_per_second": 52.3,
      "p50_itl_ms": 18, "p95_itl_ms": 42, "p99_itl_ms": 67,
      "tpot_ms": 19.1, "max_itl_ms": 89,
      "prompt_tokens": 22, "completion_tokens": 523,
      "cost_usd": 0.00528,
      "cold_start_ms": 1820, "steady_state_tps": 54.1,
      "itl_samples": 522, "timestamp": "2026-06-09T18:30:00+00:00"
    }
  ]
}
```
