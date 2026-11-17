# model module

把 LLM 调用从 core 里搬出来：core 只改一行 `provider = "module"`，对话行为
与内置 OpenAI 适配器一致。模块本身只做一件事——`chat`——**不执行工具**，
只把模型的 `tool_calls` 原样（且形状正确）交回 core。

```
core  ModuleLLMProvider ──router──▶ model/chat ──HTTP──▶ OpenAI 兼容上游
                                     │
                                     └─ 返回 message + usage + finish_reason（非流式）
```

## 工具

| 工具 | 签名 | risk | timeout | idempotent |
|---|---|---|---|---|
| `chat` | `chat(messages, tools=[], temperature=0.7, model="")` | low | 120 s | **false** |

`concurrency = "exclusive"`，`priority = 100`（SDK 允许的上限；core 在多个
model 模块中取最高，FR-6.3）。`exclusive` 的原因不是性能，是账单：两个实例
同时服务同一轮会重复计费，而 completion 不是幂等的。

`idempotent=false` 是刻意的：router 不会替我们重试。模块内部只对**尚未读到
任何响应**的 429/5xx/连接错误重试（见下），那不会重复计费。

`model` 是可选参数，M6 §四 的表里没有它，但 core 的 `ModuleLLMProvider` 在
turn 被 pin 到某个模型时会发 `model`（`core/vade/llm/module.py`）。schema 用
`additionalProperties: false`，不接受它会让 pinned turn 直接
`INVALID_ARGUMENT`，所以这里留一个可选通道：空串 = 用 `MODEL_NAME`。

## 返回契约（M6 §五）

```json
{"ok": true, "result": {
  "message": {"role": "assistant", "content": "...", "tool_calls": [
    {"id": "call_1", "type": "function", "function": {"name": "fs__read_file", "arguments": "{\"path\":\"/a\"}"}}
  ]},
  "usage": {"prompt_tokens": 12, "completion_tokens": 34},
  "finish_reason": "stop"
}}
```

几条硬规则，全部由测试钉住：

- `tool_calls[].function.arguments` **一定是字符串**。上游给对象（Anthropic 的
  `input`、某些兼容层的 `arguments: {...}`）时，这里 `json.dumps(..., sort_keys=True)`
  成字符串；core 按字符串解析，给对象会静默丢调用。
- `message.role` 固定 `assistant`；`content` 为 `null` 时变成 `""`（不是 `null`）。
- `message.tool_calls` 永远是数组（没有调用就是 `[]`）。
- `usage` 只有 `prompt_tokens` / `completion_tokens` 两个整数字段，缺失为 `0`；
  上游的 `input_tokens` / `output_tokens` / `total_tokens` / `*_details` 不出现。
- `finish_reason` 会归一：`end_turn`/`stop_sequence`/`eos` → `stop`，
  `tool_use` → `tool_calls`，`max_tokens` → `length`；缺失时按有无 tool_calls
  补 `tool_calls` / `stop`。
- 供应商标记（`system_fingerprint`、`x_vendor`、`function.parsed` …）**一个都
  不透传**。
- `truncated: true` 只在内容被 48 KiB 预算截断时出现（见“结果预算”）。

## 配置

全部来自环境变量，**每次调用重新读取**（换 key / base URL 不需要重启模块）。

| 变量 | 默认 | 含义 |
|---|---|---|
| `MODEL_BASE_URL` | `https://api.openai.com/v1` | 上游根地址；也可直接给 `.../chat/completions` |
| `MODEL_NAME` | `gpt-4o-mini` | 调用未指定 `model` 时使用 |
| `MODEL_API_KEY_ENV` | `MODEL_API_KEY` | **存放 key 的环境变量名** |
| `MODEL_API_KEY` | 未设置 | key 本体；只从这个（或上面改名的）变量读 |
| `MODEL_TIMEOUT_S` | `120` | 单次尝试的 socket 超时，`0.1…600` |
| `MODEL_MAX_RETRIES` | `2` | 首次失败后的重试次数，`0…5` |
| `MODEL_RETRY_BASE_S` | `0.5` | 退避基数，`0…10` |

key 的规则：

- 只从环境读，**不**出现在 `manifest.json`、配置文件、日志、错误消息里。
- key 未设置时不发 `Authorization` 头（本地 llama.cpp / vLLM 常常不需要），
  而不是发一个空 Bearer。
- 配错的 `MODEL_BASE_URL` / 超时 / 重试次数 → `INTERNAL`，消息里只有变量名和
  允许范围，没有 key。

## 请求与上游形状

请求体（`stream` 显式 `false`，不依赖供应商默认）：

```json
{"model": "...", "messages": [...], "temperature": 0.7, "stream": false, "tools": [...]}
```

`tools` 只在非空时出现。接受的响应形状：

| 形状 | 处理 |
|---|---|
| OpenAI `choices[0].message` | 标准路径 |
| 扁平 `{message: {...}, usage: ...}` | 当作 choice |
| Anthropic 风格 `{content: [{type:text}, {type:tool_use}], stop_reason}` | 文本拼接、`tool_use` 转成 OpenAI tool call |
| `delta` 而不是 `message`（个别兼容层） | 当作 message |
| 200 + `error` 对象 / 非 JSON / 非对象 | `INTERNAL`，不回显上游 message |

## 重试与超时

| 上游结果 | 重试 | 最终错误码 | retriable |
|---|---|---|---|
| 429 | 最多 `MODEL_MAX_RETRIES` 次 | `OVERLOADED` | true |
| 408 | 同上 | `TIMEOUT` | true |
| 5xx | 同上 | `UNAVAILABLE` | true |
| 连接失败（`URLError` / `OSError`） | 同上 | `UNAVAILABLE` | true |
| 其它 4xx | **从不** | 400/415/422 `INVALID_ARGUMENT`，401/403 `PERMISSION_DENIED`，404 `NOT_FOUND`，413 `PAYLOAD_TOO_LARGE` | false |
| socket 超时 | **不重试** | `TIMEOUT` | true |

- 退避：第 n 次失败后等 `MODEL_RETRY_BASE_S * 2^(n-1) + uniform(0, MODEL_RETRY_BASE_S)`。
  默认第一次约 `0.5…1.0 s`，第二次约 `1.0…1.5 s`。`Retry-After` 头优先，最大 10 s。
- `MODEL_MAX_RETRIES=2` 的意思是**最多 3 次 HTTP 尝试**。
- 超时不重试：一次尝试可能吃掉整个 120 s 工具预算，再试一次只会保证 SDK 的
  `asyncio.wait_for` 先杀，而不是我们给出结构化的 `TIMEOUT`。core 可以据此
  决定“新开一轮”。
- 总预算是 **118 s**（120 s 工具 deadline 留 2 s 序列化错误）；单次尝试的超时是
  `min(MODEL_TIMEOUT_S, 剩余预算)`。默认 `MODEL_TIMEOUT_S=120` 时第一次尝试实际
  只有 118 s，因此**想让模块内重试有意义，请把 `MODEL_TIMEOUT_S` 调到 30 左右**：

  ```bash
  MODEL_TIMEOUT_S=30 MODEL_MAX_RETRIES=2   # 最多 3×30 s + 退避，仍在 118 s 内
  ```

真实的两次 429 重试日志（`retry_base_s=0.2`，含抖动；`ts` 由 SDK 的 JSON
logger 打）：

```json
{"ts":"2026-11-17T05:29:10.470Z","level":"warning","module_id":"com.example.model","instance_id":"main","msg":"model retry","tool":"chat","attempt":1,"status":"http 429","delay_ms":252,"max_attempts":3}
{"ts":"2026-11-17T05:29:10.731Z","level":"warning","module_id":"com.example.model","instance_id":"main","msg":"model retry","tool":"chat","attempt":2,"status":"http 429","delay_ms":436,"max_attempts":3}
```

## 参数校验（M6 §四）

| 检查 | 上限 | 失败 |
|---|---|---|
| `messages` 条数 | 1…200 | `INVALID_ARGUMENT` |
| `role` | `system` / `user` / `assistant` / `tool` | `INVALID_ARGUMENT` |
| `content` | string 或 null（v1 纯文本） | `INVALID_ARGUMENT` |
| 单条消息（紧凑 JSON 字节） | 64 KiB | `INVALID_ARGUMENT` |
| `messages` 总量（紧凑 JSON 字节） | 512 KiB | `INVALID_ARGUMENT` |
| `tools` | 数组，≤128 个对象 | `INVALID_ARGUMENT` |
| `temperature` | `[0, 2]`，拒绝 bool | `INVALID_ARGUMENT` |
| `model` | 字符串，≤200 字符 | `INVALID_ARGUMENT` |

校验发生在**打开 socket 之前**。注意 v1 传输一帧明文上限是 65519 字节且没有
分片，所以真正跨 core 到达模块的请求会先撞上 frame 上限；512 KiB 这道检查是
给 `--standalone` 和未来分片留的，不是“线上能收到 512 KiB”。

## 结果预算 48 KiB

一个结果 JSON（`json.dumps(ensure_ascii=False, separators=(",", ":"))` 的 UTF-8
字节数）超过 **49152** 时：

1. 从头保留 `message.content` 的最长前缀，使整个结果重新落回 48 KiB；
2. 置 `result.truncated = true`（**只在截断时出现**）；
3. `usage` / `finish_reason` / `tool_calls` 原样保留。

**不截断 tool_calls 的 arguments**：那会产生非法 JSON；**不丢 tool_calls**：那是
静默跳过动作。如果清空 content 后 tool_calls 本身就超过预算，返回
`PAYLOAD_TOO_LARGE`，并在错误里说清楚，而不是假装成功。

SDK 的信封里也有一个 `truncated`（`call_tool` 返回的），v0.1 永远是 `false`——
它描述的是传输层，不描述本模块的内容预算。两者不要混淆。

48 KiB 是 65519 字节一帧里留给结果的份额，剩余约 16 KiB 给 JSON-RPC 信封、
`duration_ms` 与 Noise tag。

## v1 不流式：首字延迟 = 整段生成时间

这是本模块最重要的使用限制。

- 模块发 `stream: false`，拿到完整 response 才返回；
- core 的 `ModuleLLMProvider` 把它作为**一个 `Delta`** 推给 loop，UI 只收到一个
  `turn.delta`，然后 `turn.completed`；
- 因此**首字延迟等于整段生成时间**。内置适配器是 SSE 流式，用户 300 ms 后就能
  看到第一个字；模块模式下，一个 20 秒的长回答会让界面空白 20 秒，然后整段出现。
  没有打字机效果，也没有“正在生成”的中间文本。
- 对短回答（工具调用、一句话确认）影响很小；对长文生成、代码生成影响明显。

缓解手段（按优先级）：

1. 上游限制 `max_tokens`（例如 1024），把最坏等待压到可接受范围；
2. `MODEL_TIMEOUT_S=30`，避免一次调用占满 120 s；
3. v2 再做 SSE：模块侧解析 `data:` 行并通过 SDK 的流式能力转发——**v1 不做**，
   因为协议层没有增量结果通道，硬做只能在模块里缓冲，等于假流式。

## 日志与隐私

每次调用一行（`stderr`，JSON lines）：

```json
{"ts":"...","level":"info","module_id":"com.example.model","instance_id":"main","msg":"model chat","tool":"chat","messages":1,"messages_digest":"4f2c…","prompt_tokens":12,"completion_tokens":34,"finish_reason":"stop","attempts":1,"truncated":false,"duration_ms":842}
```

- `messages_digest` = 规范化 messages 数组（`sort_keys=True`、紧凑分隔符）的
  sha256 **前 16 位十六进制**。同样的 prompt 得到同样的 digest，可以关联工单，
  但日志里没有原文。
- 成功记 `messages` / `messages_digest` / `prompt_tokens` / `completion_tokens` /
  `finish_reason` / `attempts` / `truncated` / `duration_ms`；
- 失败记 `code` / `attempts` / `messages` / `messages_digest` / `duration_ms`；
- 重试记 `attempt` / `status` / `delay_ms` / `max_attempts`。
- **绝不记 prompt、completion、tool 参数、API key**。错误消息只带 HTTP 状态和
  上游 `error.type` / `error.code`，不带上游 `error.message`（它可能回显输入）。
- `tests/test_server.py` 用一段标记字符串对 stdout/stderr 做子串断言；日志里出现
  原文即测试失败。
- HTTP 层**不跟随重定向**（避免把 `Authorization` 带到另一个 host），响应体上限
  2 MiB；越界都是 `PAYLOAD_TOO_LARGE`，3xx 是 `INTERNAL`。

## 与内置适配器一致

`tests/test_builtin_parity.py` 把同一份 OpenAI 响应同时喂给：

- 本模块的 `normalize_response`；
- core 内置适配器的 `vade.llm.openai_compat.parse_chunk`（纯函数）。

断言 `content`、`(id, name, arguments)` 三元组、`finish_reason` 完全相等，
`usage` 与 `flatten_usage` 的两个计数字段相等。core 不在 `sys.path` 时该文件
自动 skip，不影响模块独立测试。

```bash
python -m pytest tests/test_builtin_parity.py -q     # 3 passed（需 core 检出）
```

形状差异为**空**：M6 §五 的 `message` 就是内置适配器消费的
`Delta.text` + `Delta.tool_calls` + `usage` + `finish_reason` 的 JSON 投影。

## 布局

```
.
├── manifest.json                 generated — never hand-edited
├── pyproject.toml
├── src/model_module/
│   ├── tools.py                  纯函数：校验 / 请求体 / 规整 / 预算；零 SDK import
│   ├── provider.py               OpenAI 兼容客户端（stdlib urllib）+ 重试/超时
│   ├── server.py                 ModuleApp + ToolSpec + 日志；无业务逻辑
│   └── __main__.py               python -m model_module
├── tests/test_tools.py           无 SDK 也能跑
├── tests/test_provider.py        假 transport，无 socket
├── tests/test_server.py          SDK 不在时 skip；含本地 http.server 的 CLI 测试
├── tests/test_builtin_parity.py  core 不在时 skip
├── Dockerfile
└── data/                         模块 key 落这里（gitignored，容器要挂卷）
```

`tools.py` 不 import `vade_sdk`（AST 测试钉住），`server.py` 不碰 HTTP、不做
校验、不做响应整形。签名就是 schema，`manifest.json` 由同一条生成路径产出：

```bash
PYTHONPATH=src python -m model_module --print-manifest > manifest.json
PYTHONPATH=src python -m model_module --print-manifest | diff - manifest.json   # 空
```

## 运行

```bash
pip install -e /path/to/Vade/sdk/python     # 或 pip install "vade-sdk==0.1.*"
pip install -e .

pytest tests -q                              # 94 passed（含 core parity；无 SDK 时 server/parity skip）
python -m model_module --check               # exit 0，manifest 打到 stdout
python -m model_module --standalone chat \
  --args '{"messages":[{"role":"user","content":"用一句话说明什么是模块化"}]}'
```

接本地 OpenAI 兼容服务（无需 key）：

```bash
MODEL_BASE_URL=http://127.0.0.1:8080/v1 MODEL_NAME=qwen2.5 MODEL_TIMEOUT_S=30 \
  python -m model_module --standalone chat \
  --args '{"messages":[{"role":"user","content":"hi"}]}'
```

接 core（需要 key 白名单与 core 的 Noise 公钥）：

```bash
CORE_PUBLIC_KEY=<64 hex> MODULE_KEY_PATH=data/module.key \
  MODEL_BASE_URL=https://api.example.com/v1 MODEL_NAME=... MODEL_API_KEY_ENV=MODEL_API_KEY \
  python -m model_module --core wss://127.0.0.1:8443
```

## core 侧切换

```toml
[llm]
provider = "module"          # builtin | module
model_module = "com.example.model"
fallback_to_builtin = false  # 模块不在线时是否回落
```

- `ModuleLLMProvider.select()` 按 `(-priority, -last_frame_at)` 选；本模块
  `priority=100`，另一个 50 的模块不会被选中。
- 模块不在线且 `fallback_to_builtin=false` → `turn.failed(UNAVAILABLE)`，错误
  消息是 "no model module is online"（可读，不是 traceback）。
- `fallback_to_builtin=true` → 静默回落到内置适配器，会话不中断，但这次 turn 的
  `llm.call` 归因是 builtin。
- 切换不改历史：transcript 里已有的 assistant/tool 消息原样进入下一次请求。

> **本仓库未执行 core e2e。** 这里没有 core 的 router/registry 依赖；验收第 6、7
> 条（`provider=module` 完成一轮含工具调用的对话、两个 model 模块选 100）需要在
> monorepo 里带 core 跑。模块侧已把 `priority`、`exclusive`、`fallback` 依赖的
> 错误码（`UNAVAILABLE`）和工具名 `chat` 固定下来。

## 容器

**这个镜像必须从 Vade monorepo 根构建**：拆分仓没有 `sdk/python`，而
`--standalone` 也要经过 `ModuleApp`，所以只有 monorepo 布局能成立。

```bash
# 在 Vade 仓库根（把本仓源码放到 modules/model/）
docker build -f modules/model/Dockerfile -t vade/model:dev .

docker run --rm \
  -e MODEL_BASE_URL=http://host.docker.internal:8080/v1 \
  -e MODEL_NAME=qwen2.5 \
  -e MODEL_TIMEOUT_S=30 \
  vade/model:dev --standalone chat \
  --args '{"messages":[{"role":"user","content":"hi"}]}'
```

镜像以 uid 10001 运行，`/data` 是唯一可写卷，`CMD` 是
`--core wss://core:8443`。Noise 绑定 `vade_protocol` 不在镜像里：要么从带
`protocol/core` 的 monorepo 重建，要么把 `vade_protocol.py` +
`libvade_protocol.so` 挂到 `/opt/vade-sdk`。

## 五道门（本仓库能验的部分）

| 门 | 状态 |
|---|---|
| 独立运行 | `--standalone chat` 对本地 `http.server` 的测试通过（真实 urllib + JSON） |
| 独立测试 | `pytest tests -q` → 94 passed；无 SDK 时 server/parity 自动 skip，纯函数与 provider 仍然全绿 |
| 独立部署 | Dockerfile 提供；构建需要 `python:3.12-slim` 与 `websockets`/`cryptography` 轮子 |
| 独立失败 | 模块无持久状态；进程死掉后 core 侧是 `UNAVAILABLE`（由 C3 的 provider 决定） |
| 独立观测 | 每行 JSON 带 `module_id` / `instance_id` / `tool` / `duration_ms`；日志无原文 |

## 已知边界

- **非流式**：首字延迟 = 整段生成时间（见上）。
- **48 KiB 截断**：只保留 content 前缀，尾部丢失；`truncated=true` 是唯一信号。
  需要完整长文时请让上游限制长度，或等 v2 流式。
- **512 KiB 校验 vs 65519 字节 frame**：线上请求先撞 frame 上限。
- **纯文本 content**：图片/多模态 parts 被 `INVALID_ARGUMENT` 拒绝，而不是静默
  转发给一个可能不支持的供应商。
- **模块内不执行工具**：`tool_calls` 只是返回给 core；执行、审批、审计都在 core。
- **同步客户端**：SDK 在线程池里跑它；一次调用占一个线程，`exclusive` 下最多
  `max_concurrency=4` 个并发。要更高吞吐应增加实例，但那会破坏 `exclusive` 的
  单实例计费假设，v1 不做。
