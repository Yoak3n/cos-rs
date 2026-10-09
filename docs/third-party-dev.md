# cos 第三方接口开发文档

> 适用对象：想为 cos 编写插件（工具 / 服务 / LLM Provider / Agent / Shell / RPC Provider / 不变式）
> 的第三方开发者，以及想通过外部程序（IDE / 自定义 UI / 脚本）调用 cos 的集成方。
>
> 配套文档：`PLAN.md`（实施计划）、`docs/decisions.md`（设计决策）、`docs/b-abi.md`（B 形态 FFI 契约）、
> `docs/rpc.md`（stdio JSON-RPC 协议）、`docs/memory-plugin.md`（记忆插件设计）。

---

## 1. 项目概览与依赖纪律

cos 是一个「一切皆插件」的 Rust agent 主框架（思想借鉴 dsh / cordis）：

- **Context 服务仓库**：`provide` / `get` 依赖注入（类型化 TypeMap，同名唯一）；
- **事件总线**：五种分发模式（`emit` / `parallel` / `serial` / `bail` / `waterfall`）；
- **可逆效果**：`apply` 内注册的一切（服务 / 监听器 / 工具）挂进 `Fiber`，卸载逆序回滚（RAII）；
- **会话日志是唯一事实源**：模型可见的一切都先写日志（`SessionEvent` 追加式、JSONL 落盘）；
- **turn/step 主循环**：`cos-agent-loop` 驱动，工具调用走 `cos-tools` 管线；
- **启动装配**：`cordis.yml` 声明插件树，loader 按 `inject` / `provide` 建图拓扑排序后依次 `apply`。

### 依赖方向铁律

```text
plugins/* 与 cos-agent-loop
  只依赖各接缝的 Definition crate（cos-core / cos-llm / cos-tools / cos-agent /
  cos-shell / cos-session），不得开启 `cos-llm` 的 `adapters` feature（内置适配器族，
  随 feature 引入 reqwest/tokio；只有 Provider 封装插件开它）
  或 cos-agent-loop 本身。
  例外：Provider 封装插件（plugin-opencode）的职责就是把对应 Provider crate
  接进运行时，允许依赖它。

cos-core  不依赖任何上层 crate（仅类型与 futures 组合子）。
cos-contract  零运行时依赖（仅 serde），是 B 形态插件的唯一契约。
```

违反铁律的后果：编译期不报错、运行期架构腐化（具体 Provider 无法替换、循环依赖、装载顺序失控）。
CI 用 `cargo deny` 之外的 crate 依赖审查由 code review 把关。

---

## 2. crate 地图：每个 crate 的作用

### 2.1 `crates/` —— 核心与接缝

| crate | 作用 | 关键内容 |
| --- | --- | --- |
| **cos-core** | 插件化内核（地基，零上层依赖） | `Context`（provide/get、on/emit 等五分发）、`Plugin` trait、`Validate`、`Service`、`Fiber`/`EffectHandle`（RAII 逆序卸载）、`ScopeKey`/`ScopeTarget`、`CoreError` |
| **cos-contract** | B 形态版本化契约 crate（地基，零运行时依赖） | B-ABI 版本协商（`ContractVersion::compatible_with`）、`HostApi` 能力函数表（`#[repr(C)]`）、插件导出入口符号、错误码、插件清单 JSON。设计见 `docs/b-abi.md` |
| **cos-loader** | 装配器：`cordis.yml` 解析 → 工厂解析 → 拓扑排序 → 挂载 | `Profile`/`Entry`、`resolve_factory`（唯一入口）、`plugin!` 宏（inventory 静态收集）、`DlopenPlugin`（运行时 dlopen）、`dump_plan`/`--dump-config` |
| **cos-session** | 会话日志（唯一事实源） | `SessionEvent` 封闭枚举 + `Custom` 逃生舱、`Session`（append / `derive_messages`）、JSONL 持久化（`load_jsonl`/`save_jsonl`）、`TodoItem`、`ToolError`、`TurnEndReason` |
| **cos-llm** | LLM 接缝（Definition；`adapters` feature 附带内置适配器族） | `Role`/`Message`/`ContentBlock`（Text/Thinking/ToolUse）、`UserMessage`（含 `images`、排队 `id`）、`LlmRequest`、`StreamChunk`/`ChunkDelta`（含 Finish）、`LlmAdapter` trait、`LlmStream`、`LlmRegistry`（服务 `"llm"`，providers/chains/后备链、`register_factory` 程序化注册）；**`adapters` feature**（默认关）：内置适配器族——`openai`（`/chat/completions`）、`anthropic`（`/messages`）、`responses`（`/responses`），SSE 流式优先、可重试失败自动非流式兜底；`build_with_style` 按 `api_style` 分发（`register_api_style` 可注册自定义风格），`fetch_models` 拉取 Provider 模型清单（原独立 crate 已并入，P9）——由封装插件（plugin-opencode-provider / plugin-deepseek-provider / plugin-custom-provider）开启并注册为各自的 kind |
| **cos-test-support** | 测试支持（**仅 dev-dependencies 引用，不进正式二进制**） | `MockAdapter`/`MockReply`（原 cos-llm-mock 的脚本化 mock 桩）+ `ScriptedChatServer`（本地回环 `chat/completions` 服务器，e2e 经 `--llm-*` 以真实适配器协议离线驱动 CLI 链路）。**不注册任何工厂/插件条目** |
| **cos-tools** | 工具注册表 + 执行管线（Definition） | `Tool` trait、`ToolRun`/`ToolOutcome`、`ToolGuard`（单调守卫）、`ToolRegistry`（服务 `"tools"`）；管线：`tools/pre-execute`(waterfall) → 守卫 → `tools/execute`(waterfall) → 工具体 → `tools/post-execute`(waterfall) → 结果 |
| **cos-system-prompt** | prompt 段 + 工具 schema 装配 | `PromptSections`：system 提示按段组装（每段 `PromptSection { name, order, text }`，**段序显式**、按 `order` 升序渲染；段名/段序重复或空段一律拒收）、工具 schema 注入 |
| **cos-agent** | Agent 接缝（Definition） | `AgentTrait`（id/options/session/ctx/status/send/followup/steer/inject/cancel/when_idle/run_maintenance…）、`AgentRegistry`（服务 `"agents"`）、`Inbox` 双队列、task-local 因果链（`current_initiator`/`with_initiator`）、`AgentFactory` |
| **cos-agent-loop** | turn/step 驱动器，实现 `cos-agent`（宿主核心） | `wake_driver → kick → turn → agent/pre-step`(waterfall) → `step`（`agent/request`(waterfall) → LLM stream → chunk 逐条 → 工具调用）→ `turn/end`；**每步先写日志再行动** |
| **cos-shell** | shell 接缝 + local 实现（Definition） | `Shell` trait（`run`）、`ShellProvider`（服务 `"shell"`）、`LocalShell`（`cmd /C` 前台执行） |
| **cos-memory** | 记忆内核 | 关系层记忆（关系卡/账本/事件/承诺/自史），`apply_turn` 提取→编号→消解→合并，遗忘曲线删除，召回；`MemoryStore` |
| **cos-rpc** | RPC 协议引擎（与 pi 对齐） | `serve_rpc` JSONL 服务循环、命令分发、流式事件投影、`RpcProvider` trait、`RpcProviderRegistry`（服务 `"rpc-providers"`）。协议见 `docs/rpc.md` |
| **cos-invariants** | 不变式注册表 | `SessionInvariant` trait、`InvariantRegistry`（服务 `"invariants"`）；内置：seq 单调连续、turn/step 配对、`tool/call`↔`tool/result` 按 call_id 配对、模型可见 ⊆ 已记录 |

### 2.2 `plugins/` —— A 形态插件（workspace crate）

| crate | 作用 |
| --- | --- |
| **plugin-todo** | `todo_write` 工具（session 态清单，整表替换、最后写入胜出；提供 `"todo-store"` 服务）——**第三方工具插件的范本** |
| **plugin-bash** | `bash` 工具（经 `cos-shell` 接缝前台执行命令） |
| **plugin-llm** | LLM 提供商统一管理：yaml `providers`/`chains` → `LlmRegistry`；`${ENV_VAR}` 展开 |
| **plugin-opencode-provider** | **Provider 封装插件（运行时声明式装配）**：apply 时把 `"opencode"` 工厂与**套餐默认端点**（`config.plan: go\|zen`，base_url 可覆盖）+ **go/zen 内置模型目录**（每模型标注 api style：openai/anthropic/responses）注册进 `LlmRegistry`（`register_factory_with_catalog` + [`cos_llm::build_with_style`]——provider 条目的 `base_url`/`api_style` 可省略）。**新 Provider 的范本**：新 crate（适配器实现）+ 新 plugin（apply 注册工厂）+ 锚点 |
| **plugin-custom-provider** | **自定义 Provider 插件（纯配置，无需写代码）**：注册 `kind: custom`（复用 OpenAI 兼容适配器），`config.defaults` 可下沉公共字段（base_url/api_key，支持 `${ENV_VAR}`）——任意 OpenAI 兼容端点开箱即用 |
| **plugin-memory** | 关系层记忆插件（提供 `"memory"` 服务 + `remember`/`recall`/`inventory`/`demote` 四工具；LLM 未就绪时软降级禁用，不阻塞会话） |
| **plugin-rpc** | 向宿主注册默认 RPC provider（`--rpc` 委托；yml 未声明时宿主回退内置实现） |
| **plugin-todo-dlopen** | **B 形态（cdylib）** todo 薄壳插件试点（P8） |

### 2.3 `src/` —— cos CLI 宿主

- `assemble`：内置服务（tools / system-prompt / invariants / shell / agents / llm / rpc-providers）+ LLM 注册表 + 插件树 + 主 agent；
- `finish`：不变式校验 + 会话 digest + JSONL 落盘 + 优雅逆序卸载；
- 三种形态共用同一装配：REPL（默认）、`--rpc`（stdio JSON-RPC）、`--prompt`（一次性）；
- `plugins::builtin_plugin_ids`：对插件 crate 的显式引用锚点，保证其 inventory 静态注册表被链接进可执行文件。**新插件若要在宿主内默认可用，需要在这里（或依赖方）保持引用。**

---

## 3. 核心概念与关键 API（写给插件作者）

### 3.1 Plugin trait（cos-core）

```rust
pub trait Plugin: Send + Sync {
    fn id(&self) -> &'static str;                          // 插件 id
    type Config: DeserializeOwned + Validate;              // serde 反序列化 + 校验
    fn inject(&self) -> &'static [&'static str] { &[] }    // 依赖的服务名（插件 provide 表内）
    fn provide(&self) -> &'static [&'static str] { &[] }   // 提供的服务名（与 Service::NAME 一致）
    fn apply(&self, ctx: &Context, config: &Self::Config) -> CoreResult<()>;
}
```

- `apply` 内用 `ctx.provide(...)` / `ctx.on*(...)` / 注册表注册 挂载效果，全部自动进 `ctx.fiber()`，卸载时**逆序回滚**；
- 返回 `Err` 视为插件实例启动失败，loader fail loud；
- `inject` 校验只认**插件之间的 provide 表**，宿主内置服务（`"tools"` 等）对 `inject` 不可见——依赖宿主服务时在 `apply` 里 `ctx.get` 并 fail loud。

### 3.2 服务注册表

```rust
pub trait Service: Send + Sync + 'static {
    const NAME: &'static str;   // 服务名 = 注册键，与 Plugin::provide 声明一致
}

// Context
pub fn provide<T: Service>(&self, value: T) -> CoreResult<EffectHandle>; // 同名重复 → DuplicateService
pub fn get<T: Service>(&self) -> CoreResult<Arc<T>>;                     // 按类型取（TypeId 键）
```

### 3.3 事件总线（五种分发）

```rust
// 注册（挂进 fiber，卸载自动注销）
ctx.on("my/event", |payload| { ... });                       // emit：同步逐个，返回值忽略
ctx.on_parallel("my/event", |payload| BoxFuture<CoreResult<()>>);
ctx.on_serial("my/event", |payload| BoxFuture<CoreResult<Option<EventPayload>>>); // 首个 Some 即 bail 值
ctx.on_bail("my/event", |payload| -> Option<EventPayload>);  // 首个非 None 停止
ctx.on_waterfall::<P, V>("my/event", |decision| BoxFuture<'_, V>);  // 包裹剩余链，不调 next 即短路

// 分发
ctx.emit("my/event", payload);            // 同步广播
ctx.parallel(...).await;                  // 全并发；任一失败 → 聚合错误
ctx.serial(...).await;                    // 异步按序
ctx.bail("my/event", payload);            // 同步按序取首个
ctx.waterfall::<P, V>(name, initial, default).await;  // 链尾为调用方提供的默认行为
```

约定：`EventName = &'static str`，`EventPayload = Arc<dyn Any + Send + Sync>`（监听器内 downcast，D1）。

### 3.4 内置事件清单

**会话日志事件**（`SessionEventData`，先写日志再行动，模型可见 ⊆ 已记录）：

| 事件 | 说明 |
| --- | --- |
| `turn/start` / `turn/end` | turn 生命周期（含 `TurnEndReason`） |
| `step/start` / `step/end` | step 生命周期 |
| `user/message` | 用户消息 |
| `assistant/chunk` | 流式增量（token 级回放保真） |
| `assistant/message` | 完整 assistant 消息 |
| `tool/call` / `tool/result` | 工具调用与结果（按 `call_id` 配对） |
| `Custom { name, data }` | 第三方事件逃生舱（原样透传） |

**总线扩展事件**（waterfall 挂点，第三方插件可介入）：

| 事件 | 分发 | 载荷 → 返回 |
| --- | --- | --- |
| `agent/pre-step` | waterfall | `PreStepPayload` → `PreStepDecision`（每 turn 第一步，可 veto/替换） |
| `agent/request` | waterfall | `LlmRequest` → `LlmRequest`（请求前改写：记忆注入、上下文压缩等） |
| `tools/pre-execute` | waterfall | `ToolRun` → `PreDecision`（Allow / Deny(reason)） |
| `tools/execute` | waterfall | `ToolRun` → `ToolOutcome`（替换工具体实现） |
| `tools/post-execute` | waterfall | `ToolOutcome` → `ToolOutcome` |
| `agent/created` / `agent/disposed` | emit | agent 生命周期广播 |
| `agent/status` / `agent/error` | emit | 状态变更 / 错误广播 |
| `tools/result` | emit | `ToolResultPayload` 实时通知（loop 写 tool/result 日志后发出） |

**自定义事件**：两层机制，均可由第三方自由定义——

- **总线事件**（运行时开放，决策 D1）：`EventName = &'static str`（任意字符串）+ `EventPayload = Arc<dyn Any + Send + Sync>`。插件可 `ctx.emit("my/event", Arc::new(MyPayload))` 广播新事件，也可用 `on*`/waterfall 挂到既有管线（`agent/request`、`tools/*`）上介入；B 形态插件同样可 emit/on，但载荷限 JSON；
- **会话日志事件**（决策 D4）：`SessionEventData::Custom { name, data }` 逃生舱——自定义事件写入会话日志（唯一事实源），`derive_messages` 原样透传（模型可见），不破坏「模型可见 ⊆ 已记录」不变式。

命名规范：事件名用命名空间前缀防碰撞（如 `dlopen/todo-ready`、`memory/*`）。

### 3.5 LLM 接缝（cos-llm）

```rust
pub trait LlmAdapter: Send + Sync {
    fn id(&self) -> &str;
    fn input_content(&self) -> &[InputContent] { &[InputContent::Text] }  // 能力标注（text/image）
    fn stream(&self, request: &LlmRequest) -> LlmStream;   // Pin<Box<dyn Stream<Item=Result<StreamChunk, LlmError>> + Send>>
}
```

`LlmRegistry`（服务 `"llm"`）：`build(kind, config)` 按工厂构建、`register(id, adapter)`、
`register_chain(id, providers)`、`get(id)` / `resolve(chain_id)` / `resolve_id(id)` /
`supports(id, content)`。`FallbackAdapter` 语义：主 provider 未产出任何 chunk 即失败 → 自动切下一个（防内容重复）。

### 3.6 工具接缝（cos-tools）

```rust
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn parameters(&self) -> serde_json::Value;   // JSON Schema（对象）
    fn execute(&self, ctx: &Context, run: &ToolRun)
        -> BoxFuture<'static, Result<ToolOutcome, ToolError>>;
}
```

`ToolRun { call_id, name, arguments, turn, step }`；`ToolOutcome::ok(text)` / `ToolOutcome::error(text, ToolError{name, code})`。

---

## 4. 第三方开发指南（A 形态：Rust workspace crate，推荐）

> A 形态 = 与宿主同一 workspace 编译的 crate + `plugin!` 编译期注册（inventory 静态收集）。
> 简单、类型安全、直接复用宿主接缝类型。B 形态（独立 cdylib）见第 5 节。

### 4.1 通用步骤

1. **建 crate**：`plugins/my-plugin/`，加入根 `Cargo.toml` 的 `[workspace] members`（`"plugins/*"` 已通配）。
2. **声明依赖**：只依赖接缝 crate（`cos-core` + 需要的 `cos-tools`/`cos-llm`/`cos-agent`/`cos-shell`/`cos-session`）+ `cos-loader`（用 `plugin!` 宏）。
3. **实现 Plugin**：`id` / `type Config` / `tier`（缺省 `Other`；Provider 封装插件声明 `Provider`）/ `inject` / `provide` / `apply`；`Config` 派生 `Deserialize` + 实现 `Validate`（可默认放行）。
4. **注册**：`cos_loader::plugin!("my-plugin", MyPlugin);`（`MyPlugin` 需 `Default`；`"my-plugin"` 即 cordis.yml 里的 `name`）。
5. **配置**：在 `cordis.yml` 加条目（有依赖时由拓扑排序决定；**插件类型优先级**
   `Provider < Core < Other` 决定无依赖边节点的先后——内置 Provider 插件自动排到
   `llm` 前、`llm` 自动在 `memory` 前，yml 顺序基本无关，见 §4.4）。
6. **验收**：`cargo test --workspace`、`cargo fmt --all -- --check`、`cargo clippy --workspace --all-targets -- -D warnings`、`cos --config ... --dump-config` 查看装载计划。

### 4.2 示例一：工具插件（完整可编译骨架）

```rust
// plugins/plugin-my-tool/src/lib.rs
//! 第三方工具插件示例：my_lookup 工具。
#![warn(missing_docs)]

use std::sync::Arc;

use cos_core::{Context, CoreError, Plugin, PluginTier, Validate};
use cos_tools::{Tool, ToolOutcome, ToolRegistry, ToolRun};
use futures::future::BoxFuture;
use serde::Deserialize;

/// 插件配置（示例：可留空）。
#[derive(Deserialize)]
pub struct MyToolConfig;

impl Validate for MyToolConfig {}

/// 工具实现：对象安全、Send + Sync。
struct MyLookupTool;

impl Tool for MyLookupTool {
    fn name(&self) -> &'static str {
        "my_lookup"
    }

    fn description(&self) -> &'static str {
        "查询第三方服务并返回结果"
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "查询词" }
            },
            "required": ["query"]
        })
    }

    fn execute(
        &self,
        _ctx: &Context,
        run: &ToolRun,
    ) -> BoxFuture<'static, Result<ToolOutcome, cos_session::ToolError>> {
        let query = run.arguments.get("query").and_then(|v| v.as_str()).unwrap_or("").to_string();
        Box::pin(async move {
            // 在这里调用你的第三方服务（HTTP 等）
            Ok(ToolOutcome::ok(format!("查询结果: {query}")))
        })
    }
}

/// 插件主体。
#[derive(Default)]
pub struct MyToolPlugin;

impl Plugin for MyToolPlugin {
    fn id(&self) -> &'static str {
        "plugin-my-tool"
    }

    type Config = MyToolConfig;

    // 不 provide 服务、不 inject 依赖 → 默认空实现即可

    fn apply(&self, ctx: &Context, _config: &Self::Config) -> Result<(), CoreError> {
        let registry = ctx
            .get::<ToolRegistry>()
            .map_err(|_| CoreError::ServiceNotFound("tools"))?;
        registry.register(Arc::new(MyLookupTool))?;
        Ok(())
    }
}

cos_loader::plugin!("my-tool", MyToolPlugin);
```

```yaml
# cordis.yml
- name: my-tool
```

要点：

- 工具执行是异步的：`execute` 返回 `BoxFuture`，内部可 await 网络请求；
- 工具参数在 `ToolRun.arguments`（serde_json::Value），非法 JSON 由 loop 兜底；
- 会话态副作用请写在因果链内：`cos_agent::current_initiator()` → `agent.session().append(SessionEventData::...)`（无发起者时跳过），保证「模型可见 ⊆ 已记录」；
- 想替换某工具的默认实现：监听 `tools/execute` waterfall 并返回自己的 `ToolOutcome`。

### 4.3 示例二：服务 + 事件监听插件

```rust
use cos_core::{Context, CoreError, Plugin, Service, Validate};

/// 插件提供的服务：Clone 共享内部 Arc。
#[derive(Clone, Default)]
pub struct MyStore {
    inner: Arc<std::sync::Mutex<Vec<String>>>,
}

impl Service for MyStore {
    const NAME: &'static str = "my-store";   // 与 Plugin::provide 声明一致
}

impl Plugin for MyStorePlugin {
    // ...
    fn provide(&self) -> &'static [&'static str] { &["my-store"] }

    fn apply(&self, ctx: &Context, _config: &Self::Config) -> Result<(), CoreError> {
        // 1) 注册服务（同名重复会 fail loud）
        ctx.provide(MyStore::default())?;

        // 2) 监听事件（自动挂进 fiber，卸载自动注销）
        ctx.on("agent/created", |_payload| {
            // downcast 载荷后使用
        })?;

        // 3) 介入工具执行管线（veto 或替换）
        ctx.on_waterfall::<cos_tools::ToolRun, cos_tools::PreDecision>(
            "tools/pre-execute",
            |decision| Box::pin(async move { cos_tools::PreDecision::Allow }),
        )?;
        Ok(())
    }
}
```

### 4.4 示例三：新 LLM Provider（适配器 crate + Provider 封装插件，声明式装配）

Provider 是**运行时插件**：新适配器 = 两个 crate（实现 + 封装插件），宿主与插件树其余部分零改动。

```rust
// crates/cos-llm-myprovider/src/lib.rs —— 适配器实现（不含注册）
use std::sync::Arc;
use cos_llm::{LlmAdapter, LlmError, LlmRequest, LlmStream, StreamChunk};
use futures::stream;

/// 适配器实现（对象安全：同步方法 + boxed stream）。
pub struct MyAdapter { /* base_url / api_key / model ... */ }

impl LlmAdapter for MyAdapter {
    fn id(&self) -> &str { "myprovider" }

    fn input_content(&self) -> &[cos_llm::InputContent] {
        &[cos_llm::InputContent::Text]   // 视觉模型可声明 [Text, Image]
    }

    fn stream(&self, request: &LlmRequest) -> LlmStream {
        // 组装 HTTP 请求 → 逐 chunk 产出
        let text = format!("echo: {} messages", request.messages.len());
        Box::pin(stream::once(async move {
            Ok(StreamChunk::text(text))
        }))
    }
}

/// 工厂：kind + 配置 JSON → 已实例化适配器（由 plugin-myprovider 引用）。
pub fn build_myprovider(config: &serde_json::Value) -> Result<Arc<dyn LlmAdapter>, LlmError> {
    let _ = config; // 解析 base_url/model/api_key 等
    Ok(Arc::new(MyAdapter { /* ... */ }))
}

/// kind 常量（plugin-myprovider 与宿主锚点引用）。
pub const MYPROVIDER_KIND: &str = "myprovider";
```

```rust
// plugins/plugin-myprovider/src/lib.rs —— Provider 封装插件（参照 plugin-opencode）
use cos_core::{Context, CoreError, Plugin, PluginTier, Validate};
use cos_llm::LlmRegistry;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct MyProviderConfig;
impl Validate for MyProviderConfig {}

#[derive(Default)]
pub struct MyProviderPlugin;

impl Plugin for MyProviderPlugin {
    fn id(&self) -> &'static str { "plugin-myprovider" }
    type Config = MyProviderConfig;
    /// Provider 类型（装配优先级最高）：loader 注册前扫描按类型排序——本插件
    /// 自动排到 plugin-llm（Core）之前，yml 条目顺序无关（无需改 plugin-llm）。
    fn tier(&self) -> PluginTier { PluginTier::Provider }
    fn apply(&self, ctx: &Context, _config: &Self::Config) -> Result<(), CoreError> {
        let registry = ctx.get::<LlmRegistry>()
            .map_err(|_| CoreError::ServiceNotFound("llm"))?;
        registry
            .register_factory(cos_llm_myprovider::MYPROVIDER_KIND, cos_llm_myprovider::build_myprovider)
            .map_err(CoreError::Other)
    }
}
cos_loader::plugin!("myprovider", MyProviderPlugin);
```

> **类型声明即排序**：实现 `Plugin::tier()` 返回 `PluginTier::Provider`（或 `Core`）即可
> 参与类型优先级（Provider < Core < Other，同类型保持配置顺序；`inject` 边优先）——
> 无需维护任何标记列表。

```yaml
# cordis.yml —— 声明式装配（Provider 类型自动排到 llm 之前，顺序任意）
- name: llm
  config:
    providers:
      - { id: mine, kind: myprovider, config: { base_url: "...", model: "...", api_key: "${MY_API_KEY}" } }
    chains:
      - { id: main, providers: [mine] }
- name: myprovider
```

未声明 `- name: myprovider` 而引用 `kind: myprovider` → 装配期 fail loud，错误信息列出可用
kinds 并提示声明插件（plugin-llm 的 build 错误已带此提示）。宿主侧如需 CLI 快捷方式
（`--llm-*` 走某 kind），在 `src/lib.rs` 的 `--llm-*` 分支指定 kind 常量即可。

**默认配置下沉 + 模型目录**：封装插件可用 `LlmRegistry::register_factory_with_catalog(
kind, build, defaults, catalog)` 注册——`build(kind, config)` 时**三级浅合并**（插件级
`defaults` < 模型级 `catalog[config.model]` < 条目 config，条目覆盖）。`catalog` 条目为
[`cos_llm::ModelDefaults`] `{ model, group, defaults }`：Provider 插件内置**可用模型清单**
（如 plugin-opencode-provider 的 `BUILTIN_MODELS`——go/zen 套餐各自端点/api 风格/预算，
模型间可不同），`group` 给模型打**分组标签**（套餐）——用户配置 `group: <组>` 即可按组展开，
查询面 `available_groups` / `models_in_group`（运行时与插件 crate 纯函数双形态）。
`config.models` 可追加/覆盖（同名后者生效）。`defaults`/目录中的 `${ENV_VAR}`
由注册方在 apply 内展开（`cos_llm::expand_env`）。

**api style（把协议适配交给插件）**：注册 kind 时用 `cos_llm::build_with_style` 作
`build`——合并配置的 `api_style` 决定风格构建器（缺省 `"openai"`；内置
`openai`/`anthropic`/`responses`，路径后缀由风格决定）。**自定义风格插件**：
新 crate 实现 `fn(&serde_json::Value) -> Result<Arc<dyn LlmAdapter>, LlmError>`，
apply 里 `cos_llm::register_api_style("<风格>", build)`（或经
`ctx.get::<LlmRegistry>()?.register_api_style`），然后任何 Provider 的目录条目声明
`api_style: "<风格>"` 即按此构建——同 kind 内不同模型可各用不同风格。模型清单也可
**从 Provider 拉取**：`cos_llm::fetch_models(endpoint, api_style)`（阻塞式，apply 期
调用；plugin-opencode 的 `models_endpoint` 即此用法的范本）。

> **不想写代码？** 任意 OpenAI 兼容端点直接声明 `custom-provider` 插件 + `kind: custom`
> 条目（纯配置，见 `docs/configuration.md`）——代码级自定义（本文）与纯配置自定义是
> 两条并列的路径，后者覆盖"换个 base_url/model 就用"的场景。

### 4.5 其他接缝（概览）

| 想做 | 实现 | 挂载方式 |
| --- | --- | --- |
| 自定义 Shell | `cos_shell::Shell`（`run`），包进 `ShellProvider { inner }` | `ctx.provide(ShellProvider{...})`（服务 `"shell"`，宿主已内置 LocalShell，注意同名冲突） |
| 自定义 RPC 传输 | `cos_rpc::RpcProvider`（`serve(agent, cancel)`） | `ctx.get::<RpcProviderRegistry>()?.register(...)` |
| **自定义 Agent 驱动器**（可替换，替代/扩展 turn/step 主循环） | 实现 `cos_agent::AgentFactory`（`create(root, options) -> Arc<dyn Agent>`，可基于 `cos-agent-loop` 复用或自研驱动器） | `cos_agent::agent_factory!("<id>", build)` 注册（inventory 静态收集）；宿主 `--agent-driver <id>`（或 `COS_AGENT_DRIVER`）选择，缺省 `loop`；未知驱动 → 启动失败并列出可用驱动。宿主侧需锚点（见 `src/plugins.rs::builtin_agent_drivers`）保证注册表被链接 |
| 自定义会话不变式 | `cos_invariants::SessionInvariant`（`check(&Session) -> Vec<String>`） | `ctx.get::<InvariantRegistry>()?.register(...)`；`finish` 时自动校验 |
| 监听/改写模型请求 | 监听 `agent/request` waterfall | 记忆注入、上下文压缩（超阈值滚动摘要）等 |
| 每 turn 第一步介入 | 监听 `agent/pre-step` waterfall | 消化上一 turn 的后果、维护状态 |

> **关键组件纪律**：cos 首先是一个可用的 agent——装配期关键组件（LLM、agent 驱动）缺失
> 一律**报警退出**（错误信息给出接入/替换方式与可用清单），不静默降级。可选能力
> （记忆、RPC 等）缺失只影响对应功能。

---

## 5. B 形态插件（独立 cdylib，dlopen 加载）

适用场景：**不参与宿主 workspace 编译**的独立分发插件（闭源 / 异语言 / 动态安装）。
设计详见 `docs/b-abi.md`；P8 试点（工具/事件/效果）、P9 服务桥接、P10 清单一等
公民（清单参与依赖图 + 能力裁剪）、P11 资源生命周期（字符串驻留 + `free` 诚实
回收 + 工具自动注销）、P12 暴露面审计（validate 兑现 + 清单 api 强制 + RAII
卸载顺序修复）均已落地（`plugin-todo-dlopen` + `examples/dlopen.yml` 已可跑通）。

### 5.1 契约与版本协商

- 契约 crate：`cos-contract`（零运行时依赖），持有 `API_VERSION`（当前 `0.4.0`）；
- 兼容规则：`major` 必须相等，且插件 `minor ≤ 宿主 minor`（`ContractVersion::compatible_with`）；
- 边界载荷：**一切结构化数据 = UTF-8 JSON 字符串**；不透明指针、`c_char` 字符串、整数、`u32` 版本号是仅有的跨边界类型；字符串生命周期：宿主分配（错误缓冲）、插件只读不持有（事件 JSON）、调用返回后即失效；事件名/工具名等宿主侧驻留、随插件卸载释放（P11）；
- 插件清单 JSON（id/inject/provide/config schema）供宿主预检与拓扑排序。

### 5.2 导出符号（cdylib 必须导出）

| 符号 | 签名 | 说明 |
| --- | --- | --- |
| `cos_plugin_abi_version` | `fn() -> u32` | 返回 `API_VERSION.encode()`（`major<<16|minor<<8|patch`），宿主先握手 |
| `cos_plugin_apply` | `fn(*const HostApi, HostCtx, *const c_char, ErrorBuf, usize) -> i32` | 执行注册；失败写 error_buf（UTF-8）并返回错误码 |
| `cos_plugin_validate`（可选，P12 兑现） | `fn(*const c_char, ErrorBuf, usize) -> i32` | 配置预校验：宿主在 apply **之前**调用；非零返回 → 装载失败（fail loud，错误文本入 error_buf）；缺失 = 跳过 |

### 5.3 HostApi 能力函数表（`#[repr(C)]`，字段顺序即 ABI）

| 字段 | 类型 | 语义 |
| --- | --- | --- |
| `api_version` | `u32` | 宿主侧 `API_VERSION.encode()` |
| `get_service` | `fn(ctx, name) -> *const c_void` | 按名取服务（不透明句柄）；未注入/未注册 → 空指针（能力按 inject 裁剪） |
| `emit` | `fn(ctx, name, payload_json)` | 广播事件（同步分发） |
| `on` | `fn(ctx, name, callback, userdata) -> Handle` | 注册事件监听；`free` 注销 |
| `register_effect` | `fn(ctx, disposer, userdata) -> Handle` | 注册效果；卸载时**逆序**调用 disposer |
| `free` | `fn(ctx, handle)` | 释放 on / register_effect / register_tool 句柄（**诚实回收**，P11）：监听/效果提前注销、工具移除；未知/外来/重复句柄 = 幂等无操作（插件不得重复 free，但重复无害） |
| `register_tool` | `fn(ctx, name, description, parameters_json, execute, userdata) -> Handle` | 注册工具（0.2.0 追加）：执行时宿主把 `ToolRun` 序列化为 JSON 调 `execute`，插件把 `ToolOutcome` JSON 写入 result_buf |
| `service_call` | `fn(ctx, service, method, args_json, result_buf, result_len) -> i32` | 调用服务（0.3.0 追加，P9）：`service` 必须是 `get_service` 的返回值（身份校验，伪造/悬垂 → `InvalidHandle`）；method + args JSON → 结果 JSON 写宿主缓冲；桥调用失败 → `CallFailed`（=7），错误文本入缓冲 |

### 5.4 yml 接入

```yaml
# examples/dlopen.yml
- name: ./target/debug/plugin_todo_dlopen.dll   # ./ 或 dlopen: 前缀 → 运行时 dlopen（.so/.dll/.dylib）
  config:
    marker: target/dlopen-disposed.txt          # 示例配置：卸载时验证 disposer 链
```

### 5.4.1 第三方交付形态（cdylib + cordis.patch.yml，P13）

B 形态插件作为**第三方插件**交付 = cdylib + 自带 `cordis.patch.yml`，与用户的主
`cordis.yml` 组装成完整插件列表（对标 dsh 的 bundle 自带 patch 层；层叠规则与
patch 语义见 `docs/configuration.md` §1.5）：

```
third-party-plugin/
├── plugin.dll / plugin.so / libplugin.dylib   # cdylib（含 manifest/validate 导出）
└── cordis.patch.yml                           # 启用清单：insert 本插件的 dlopen 条目
```

```yaml
# third-party-plugin/cordis.patch.yml —— 第三方包自带
- insert:
  - name: ./third-party-plugin/plugin.dll      # 路径相对用户 cwd（或绝对路径）
    config: { ... }                            # 插件配置（${ENV_VAR} 由插件自身展开）
```

用户启用方式（任选其一，推荐主 yml `patch:` 声明——主 yml 保持"完整清单声明处"）：

```yaml
# 用户 cordis.yml —— 对象形态声明第三方 patch 层
patch: [third-party-plugin/cordis.patch.yml]
entries:
  - name: todo
```

```bash
# 或 CLI 临时 overlay（可多次，后覆盖先）
cos --config cordis.yml --patch third-party-plugin/cordis.patch.yml
```

**完整插件列表的权威输出**：`cos --config cordis.yml --dump-config` —— 与装载共用
同一合并路径，输出合并后的完整条目列表，每条目带 `source` 标注来源文件
（`cordis.yml` / patch 文件路径）。

### 5.5 B 形态插件与其他插件通信（现状）

P8/P9 已桥接的通道（实现见 `cos-loader/src/dlopen.rs`）：

| 通道 | 方向 | 现状 |
| --- | --- | --- |
| 事件 `emit` / `on` | 双向 | ✅ B 插件 `emit` 的 JSON 载荷被宿主包成 `PluginEvent(Value)` 广播到总线；`on` 回调收到宿主序列化的 JSON。⚠️ 只有 **JSON 载荷**事件对 B 插件可见（非 JSON 载荷回调收到 `"{}"`）。事件名由宿主**插件级驻留区**持有、随卸载释放（P11：不再泄漏） |
| 工具 `register_tool` | B → 宿主 | ✅ 注册进 `ToolRegistry` 与 A 插件工具同池；执行 = C 回调（`ToolRun` JSON → 插件 → `ToolOutcome` JSON 写回宿主缓冲），经模型调用间接协作。**卸载/`free` 自动注销**（P11，不留僵尸工具；重载同名工具可再注册） |
| 效果 `register_effect` | B → 宿主 | ✅ 卸载时宿主逆序调用 disposer |
| 服务 `get_service` + `service_call`（P9） | B → 宿主服务 | ✅ 按名查 [`BridgeRegistry`]（服务 `"bridges"`，宿主 `assemble` 时注册内置桥）→ 不透明句柄 → `service_call(method, args_json)` → 结果 JSON。内置桥：`tools`（`list` → 工具清单）、`llm`（`kinds` / `supports`）；第三方服务实现 `cos_core::JsonBridge` 后注册即对 B 形态开放。⚠️ 服务句柄带身份校验（伪造/悬垂 → `InvalidHandle`）；宿主状态与插件实例同生命周期，插件可自持 ctx/host 指针在工具回调内调用 |
| 服务 `get_service`（未注册桥） | B → 其他插件 | ⚠️ 未注册 `JsonBridge` 的服务不可直连（返回空指针）——这类协作仍走**事件转发**：B `emit` 请求事件（JSON）→ A 插件监听并调用服务 → A `emit` 回结果事件（JSON） |

**推荐协作路径**：需要宿主/插件服务时优先走 JSON 桥（`tools`/`llm` 已内置，自定义服务实现 `JsonBridge` 即可）；未开放桥的服务走事件转发。

---

## 6. 外部程序接口：stdio JSON-RPC（`cos --rpc`）

IDE、自定义 UI、脚本等外部程序通过 **stdin/stdout 上的 JSONL 协议**调用 cos（协议与 [pi 的 RPC 模式](https://github.com/gaianet/pi/blob/main/packages/coding-agent/docs/rpc.md) 对齐，完整规范见 `docs/rpc.md`）。

### 6.1 启动

```bash
cos --config cordis.yml --rpc [--session <id>] [--no-save]
```

`cordis.yml` 需含 `- name: rpc`（demo.yml 已含）；未声明时宿主回退内置实现。

### 6.2 帧格式

- 请求：stdin 每行一个 JSON 对象；响应：`{"id"?, "type": "response", "command": "<命令>", "success": bool, "data"?, "error"?}`；事件：处理期间 stdout 实时 JSONL 流；
- 严格 JSONL，`\n` 为唯一分隔符（输入可带尾部 `\r`，自动剥离）；`id` 可选，提供则响应原样回显；
- 事件：`agent_start` / `turn_start` / `message_start` / `message_update` / `message_end` / `tool_execution_start|end` / `turn_end` / `agent_end` / `agent_settled`（由会话日志驱动，先写日志再发事件）。

### 6.3 命令

| 命令 | 说明 |
| --- | --- |
| `prompt` | 发送用户消息（异步接受，响应先到、事件随后流式）。处理中须带 `streamingBehavior`: `"steer"` 或 `"followUp"`；响应 `data.messageId` = 排队消息 id（缺省自动生成 `m-<n>`） |
| `steer` | 排队 steering 消息（工具执行完后、下一次模型调用前送达） |
| `follow_up` | 排队后续消息（agent 处理完后送达） |
| `abort` | 中止当前操作（保留已排队消息） |
| `cancel_message` | **cos 扩展**：取消队列中指定 id 的待处理消息（已开始处理则无法取消；排队去重：重复 id 拒绝） |
| `get_state` | `{isStreaming, sessionId, sessionName, messageCount, pendingMessageCount}` |
| `get_messages` | 模型可见消息历史（pi 风格 role/content） |
| `get_last_assistant_text` | 最后一条助手文本（无则 `text: null`） |
| `get_session_stats` | `{sessionId, userMessages, assistantMessages, toolCalls, toolResults, totalMessages, tokens}` |
| `get_commands` | 命令清单 |
| `exit` | **cos 扩展**：响应后优雅退出 |

未实现命令（`set_model`、`compact`、`bash`、会话树等）返回 `success: false` + `error`，协议向前兼容。

### 6.4 示例

```json
{"id": "req-1", "type": "prompt", "message": "你好"}
{"id": "req-1", "type": "response", "command": "prompt", "success": true, "data": {"messageId": "req-1"}}

// 带图片（pi ImageContent 格式 → data URL）
{"type": "prompt", "message": "这是什么？", "images": [{"type": "image", "data": "<base64>", "mimeType": "image/png"}]}
```

---

## 7. 验收与调试

- **编译/测试**：`cargo test --workspace`、`cargo fmt --all -- --check`、`cargo clippy --workspace --all-targets -- -D warnings`；
- **装载计划**：`cos --config cordis.yml --dump-config` 输出与真实装载同序的 JSON（含插件顺序、inject/provide、dlopen 条目）；
- **fail loud**：环依赖 / 缺依赖 / 未知插件 / 重复 provide / 配置解析或校验失败，全部启动即报错（可读错误：`entry_id` + 工厂名）；
- **会话保真**：`finish` 时运行全部不变式（seq 单调、turn/step 配对、tool/call↔tool/result 配对、模型可见 ⊆ 已记录）；插件写会话日志请用 `SessionEventData` 封闭枚举或 `Custom { name, data }` 逃生舱；
- **回放**：`RunReport.events` / JSONL 落盘可完整重放一轮交互；
- **插桩点**：需要观察内部行为时优先挂 `agent/request`、`tools/*` 等 waterfall/emit 事件，不要改宿主代码。

---

## 8. 快速参考：常用服务名与注册表

| 服务名 | 类型 | 提供者 |
| --- | --- | --- |
| `"tools"` | `cos_tools::ToolRegistry` | 宿主内置（装配时 provide） |
| `"llm"` | `cos_llm::LlmRegistry` | 宿主内置空表，`plugin-llm` 按配置填充 |
| `"agents"` | `cos_agent::AgentRegistry` | 宿主内置 |
| `"shell"` | `cos_shell::ShellProvider` | 宿主内置（LocalShell） |
| `"invariants"` | `cos_invariants::InvariantRegistry` | 宿主内置 |
| `"rpc-providers"` | `cos_rpc::RpcProviderRegistry` | 宿主内置空表，`plugin-rpc` 注册默认实现 |
| `"bridges"` | `cos_core::BridgeRegistry` | 宿主内置；`tools`/`llm` 桥已注册（B 形态插件经 `get_service`/`service_call` 调用） |
| `"memory"` | `plugin-memory` 提供 | 需 yml 声明 `- name: memory` |
| `"todo-store"` | `plugin-todo` 提供 | 需 yml 声明 `- name: todo` |

宿主内置服务对插件 `inject` 不可见（inject 只认插件间 provide 表）；插件内一律 `ctx.get::<T>()` 取用，缺服务即 fail loud。
