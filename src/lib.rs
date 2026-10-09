//! cos —— 插件化主干库 + CLI 宿主。
//!
//! 三种形态共用同一装配（[`assemble`]）与收尾（[`finish`]）：
//! - [`run`]：一次性（`--prompt`）——装配 → 一轮 → 收尾，返回 [`RunReport`]；
//! - `repl::serve_repl`：交互式 REPL（命令行持续对话）；
//! - `rpc::serve_rpc`：stdio JSON-RPC 服务（每行一个请求/响应，供外部程序调用）。
//!
//! # 作为库嵌入（零插件）
//!
//! 框架核心与插件解耦：`config_path: None` 装配时**不装载任何插件**，只提供内置服务
//! （Context 事件总线 / 服务仓库 / 工具注册表 / LLM 注册表 / agent 注册表 / 会话日志）。
//! 库用户自行注册服务与工具，再经 [`AgentRegistry`] 用自研 [`LlmAdapter`] 创建 agent。
//! 完整示例见 `examples/embed.rs`；框架各层经模块别名暴露：
//! [`core`]（Context/Plugin/Service）、[`session`]、[`llm`]、[`tools`]、[`agent`]、
//! [`loader`]、[`memory`]、[`shell`]、[`invariants`]、[`rpc`]、[`contract`]。

#![warn(missing_docs)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub use cos_agent::{Agent, AgentOptions, AgentRegistry, CreateAgentOptions};
pub use cos_core::{BridgeRegistry, Context, CoreError, Plugin, Service};
pub use cos_llm::{
    ChunkDelta, InputContent, LlmAdapter, LlmError, LlmRegistry, LlmRequest, LlmStream, Message,
    StreamChunk, ToolResultMessage, UserMessage,
};
pub use cos_loader::{LoadError, LoadedApp, Profile};
pub use cos_memory::MemoryStore;
pub use cos_session::{AbortCause, Session, SessionError, ToolError, TurnEndReason};
pub use cos_tools::{Tool, ToolGuard, ToolOutcome, ToolRegistry, ToolRun};

// 框架各层（库嵌入时按需取用；模块别名 = 内部 crate 的完整再导出）。
// 注：rpc 层经 `pub mod rpc` 再导出（模块内 `pub use cos_rpc::*`），避免与 CLI 宿主模块同名。
pub use cos_agent as agent;
pub use cos_contract as contract;
pub use cos_core as core;
pub use cos_invariants as invariants;
pub use cos_llm as llm;
pub use cos_loader as loader;
pub use cos_memory as memory;
pub use cos_session as session;
pub use cos_shell as shell;
pub use cos_system_prompt as system_prompt;
pub use cos_tools as tools;

use cos_invariants::{InvariantRegistry, register_defaults};
use cos_session::{
    SESSION_FORMAT_VERSION, SessionEventData, SessionHeader, load_jsonl, save_jsonl,
};
use cos_shell::provide_local_shell;
use cos_system_prompt::{PromptSection, PromptSections};
#[cfg(feature = "plugin-opencode-provider")]
use plugin_opencode_provider::OPENCODE_KIND;

pub mod plugins;
pub mod repl;
pub mod rpc;
pub mod schema;

pub use schema::result::AppError;
pub use schema::{RunConfig, RunReport};

/// 装配结果（REPL / RPC / 一次性 / 库嵌入共用）。
pub struct Assembled {
    /// 根上下文（服务/事件总线）。
    pub root: Context,
    /// 已装载插件树（零插件装配 = 空树）。
    pub app: loader::LoadedApp,
    /// 主 agent（对话入口）；未配置 LLM 时为 `None`（库嵌入可自行注册适配器后创建）。
    pub agent: Option<Arc<dyn Agent>>,
}

impl Assembled {
    /// 主 agent；未配置 LLM 时返回引导错误（CLI 形态用；库嵌入请自行装配适配器）。
    pub fn agent(&self) -> Result<&Arc<dyn Agent>, AppError> {
        self.agent.as_ref().ok_or_else(|| {
            AppError::Other(
                "关键组件缺失：未配置 LLM（不再隐式回退演示脚本）。请任选其一：\n\
                 \x20 1) 命令行：--llm-base-url <url> --llm-model <model> --llm-api-key <key>（或 COS_LLM_* 环境变量）；\n\
                 \x20 2) cordis.yml 配置 plugin-llm 的 providers/chains（参考 examples/llm.yml）；\n\
                 \x20 3) Provider 为声明式插件：--llm-* 或 kind: opencode 均需 yml 声明 - name: opencode-provider（参考 examples/demo.yml）；\n\
                 \x20 4) 作为库嵌入：config_path: None 装配后自行注册 LlmAdapter 并经 AgentRegistry 创建 agent（参考 examples/embed.rs）。"
                    .into(),
            )
        })
    }
}

/// 一轮交互的摘要（REPL/RPC 共用）。
#[derive(Debug, Clone)]
pub struct TurnSummary {
    /// 结束的 turn 号。
    pub turn: u32,
    /// 该 turn 的助手文本（多步拼接）。
    pub reply: String,
    /// 工具轨迹（如 "todo_write → 已写入 1 条任务"）。
    pub tool_trace: Vec<String>,
    /// 是否被取消信号中断。
    pub cancelled: bool,
    /// turn 失败原因（`turn/end` 记 Error 时；None = 正常）。
    pub error: Option<String>,
}

/// 装配：内置服务 + LLM 注册表 + 插件树 + 主 agent（三种形态共用）。
///
/// LLM 解析优先级（失败即启动失败，无隐式兜底）：
/// `--agent-llm <id>` > `--llm-*` 的 "default" > yml "main"（链或提供商）> yml 恰好一个提供商。
pub async fn assemble(config: &RunConfig) -> Result<Assembled, AppError> {
    let root = Context::root();
    // 内置服务（app 装配，先于插件树）
    root.provide(ToolRegistry::new(&root))?;
    root.provide(PromptSections::new(&root))?;
    root.provide(InvariantRegistry::new(&root))?;
    provide_local_shell(&root)?;
    root.provide(AgentRegistry::new(&root))?;
    // LLM 统一管理：宿主装配空注册表；--llm-* 注册 "default"；plugin-llm 按 yml 填充
    root.provide(LlmRegistry::new(&root))?;
    // RPC 提供者注册表：plugin-rpc 的 apply 注册默认实现；未装配时 --rpc 回退内置
    root.provide(cos_rpc::RpcProviderRegistry::new())?;
    // JSON 桥（P9）：B 形态插件经 get_service/service_call 调用宿主服务
    root.provide(BridgeRegistry::new(&root))?;
    root.get::<PromptSections>()
        .expect("刚装配")
        .append(PromptSection::new(
            "persona",
            10,
            "你是 cos 演示助手，工具结果要如实汇报。",
        ))
        .map_err(|error| AppError::Other(format!("缺省 persona 段被拒：{error}")))?;
    register_defaults(&root.get::<InvariantRegistry>().expect("刚装配"));

    // 主 agent 驱动（可替换设计，agent_factory! 注册表）：--agent-driver <id>，缺省 "loop"；
    // 未知/未注册驱动 = 关键组件缺失 → 报警退出（可用驱动列表一并给出）
    {
        let agents = root.get::<AgentRegistry>().expect("刚装配");
        let driver = config
            .agent_driver
            .clone()
            .unwrap_or_else(|| cos_agent_loop::LOOP_DRIVER_ID.to_string());
        agents
            .set_driver(&driver, &serde_json::Value::Null)
            .map_err(|error| {
                AppError::Other(format!(
                    "关键组件缺失：agent 驱动 '{driver}' 不可用（{error}）。\
                     可用驱动: {}。自定义驱动 = 新 crate + agent_factory!(\"<id>\", build) 注册 + 锚点。",
                    agents.driver_ids().join(", ")
                ))
            })?;
    }

    // JSON 桥注册：内置服务对 B 形态插件开放（tools 清单 / llm 能力查询）
    {
        let bridges = root.get::<BridgeRegistry>().expect("刚装配");
        bridges.register(
            ToolRegistry::NAME,
            root.get::<ToolRegistry>().expect("刚装配"),
        )?;
        bridges.register(
            LlmRegistry::NAME,
            root.get::<LlmRegistry>().expect("刚装配"),
        )?;
    }

    // 插件树（锚定 + yml 装载，见 plugins 模块；plugin-opencode 在此注册工厂）。
    // config_path: None = 零插件装配（库嵌入：只提供内置服务，插件树为空）。
    // P13：patch 层叠（主 yml `patch:` 声明 + 同目录 cordis.patch.yml + --patch）。
    let app = plugins::load(&root, config.config_path.as_deref(), &config.patch_files)?;

    // --llm-*：opencode 快捷方式（在插件树之后——工厂由 plugin-opencode 声明注册）。
    // 未声明插件 = 关键组件缺失 → 报警退出。
    if let Some(cfg) = &config.llm {
        #[cfg(feature = "plugin-opencode-provider")]
        {
            let llm_registry = root.get::<LlmRegistry>().expect("刚装配");
            let adapter = llm_registry
                .build(
                    OPENCODE_KIND,
                    &serde_json::json!({
                        "base_url": cfg.base_url,
                        "api_key": cfg.api_key,
                        "model": cfg.model,
                        "streaming": cfg.streaming,
                        "max_tokens": 2048,
                    }),
                )
                .map_err(|error| {
                    AppError::Other(format!(
                        "关键组件缺失：--llm-* 需要 opencode Provider，但工厂不可用（{error}）。\
                         请在 cordis.yml 声明 - name: opencode-provider 插件（须在 llm 条目之前，参考 examples/demo.yml）。"
                    ))
                })?;
            llm_registry.register("default", adapter)?;
        }
        #[cfg(not(feature = "plugin-opencode-provider"))]
        {
            let _ = cfg;
            return Err(AppError::Other(
                "关键组件缺失：--llm-* 快捷方式需要 opencode Provider 插件，但本构建未启用 \
                 feature \"plugin-opencode-provider\"（default-features = false 时需显式启用）。\
                 请启用该 feature，或改用 cordis.yml 声明 - name: opencode-provider。"
                    .into(),
            ));
        }
    }

    // 主 agent：LLM 统一管理解析。无任何可用 LLM → 不创建 agent（`agent` 为 None）：
    // CLI 形态经 [`Assembled::agent`] 保持"无 LLM 启动失败"；库嵌入可自行注册适配器后创建。
    let llm_registry = root.get::<LlmRegistry>().expect("刚装配");
    let resolution = if let Some(id) = &config.agent_llm {
        let adapter = llm_registry
            .resolve_id(id)
            .map_err(|error| AppError::Other(format!("agent LLM '{id}' 不可用: {error}")))?;
        Some((adapter, Some("llm-registry".to_string()), Some(id.clone())))
    } else if config.llm.is_some() {
        let adapter = llm_registry
            .resolve_id("default")
            .map_err(|error| AppError::Other(format!("agent LLM 'default' 不可用: {error}")))?;
        Some((
            adapter,
            Some("opencode".to_string()),
            config.llm.as_ref().map(|cfg| cfg.model.clone()),
        ))
    } else if let Ok(adapter) = llm_registry.resolve_id("main") {
        // yml plugin-llm 定义了 main 链/提供商 → 自动使用（无需 --agent-llm）
        Some((
            adapter,
            Some("llm-registry".to_string()),
            Some("main".to_string()),
        ))
    } else if let Some(only) = single_provider(&llm_registry) {
        // yml 恰好定义一个非 "default" 提供商 → 自动使用
        let adapter = llm_registry
            .resolve_id(&only)
            .map_err(|error| AppError::Other(format!("agent LLM '{only}' 不可用: {error}")))?;
        Some((adapter, Some("llm-registry".to_string()), Some(only)))
    } else {
        None
    };
    let agent = match resolution {
        Some((adapter, provider, model)) => Some(
            root.get::<AgentRegistry>()
                .expect("刚装配")
                .create(CreateAgentOptions {
                    session: None,
                    session_id: config.session_id.clone(),
                    options: AgentOptions {
                        provider,
                        model,
                        max_tokens: None,
                        role: None,
                    },
                    adapter,
                })
                .await?,
        ),
        None => None,
    };

    Ok(Assembled { root, app, agent })
}

/// 跑一轮交互：followup → 等 idle（可被取消信号中断）→ 总结该 turn。
pub async fn run_turn(
    agent: &Arc<dyn Agent>,
    message: UserMessage,
    cancel: Option<&Arc<AtomicBool>>,
) -> TurnSummary {
    let before_turn = last_turn(agent.session());
    agent.followup(message);
    let mut cancelled = false;
    match cancel {
        Some(flag) => {
            tokio::select! {
                _ = agent.when_idle() => {}
                _ = wait_for_cancel(flag.clone()) => {
                    agent.cancel(AbortCause::User, false);
                    agent.when_idle().await;
                    cancelled = true;
                }
            }
        }
        None => agent.when_idle().await,
    }
    summarize_turn(agent.session(), before_turn + 1, cancelled)
}

/// 注册表中恰好一个非 "default" 提供商 → 返回其 id（零参数自动使用）。
fn single_provider(registry: &LlmRegistry) -> Option<String> {
    let mut candidates = registry
        .list()
        .into_iter()
        .filter(|(id, _)| id != "default")
        .map(|(id, _)| id);
    let only = candidates.next()?;
    if candidates.next().is_some() {
        return None; // 多于一个 → 不自动选（避免猜错意图）
    }
    Some(only)
}

/// 会话里最后一个 turn 号（0 = 空会话）。
fn last_turn(session: &cos_session::Session) -> u32 {
    session
        .events()
        .iter()
        .filter_map(|event| match &event.data {
            SessionEventData::TurnStart { turn } => Some(*turn),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

/// 从会话日志总结某 turn：助手文本 + 工具轨迹。
fn summarize_turn(session: &cos_session::Session, turn: u32, cancelled: bool) -> TurnSummary {
    let mut reply = String::new();
    let mut calls: Vec<(String, String)> = Vec::new(); // (call_id, 显示)
    let mut results: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut error = None;
    for event in session.events() {
        match &event.data {
            SessionEventData::ToolCall {
                turn: t,
                call_id,
                name,
                arguments,
                ..
            } if *t == turn => {
                calls.push((call_id.clone(), format!("{name} {arguments}")));
            }
            SessionEventData::ToolResult {
                turn: t,
                call_id,
                message,
                ..
            } if *t == turn => {
                results.insert(call_id.clone(), message.content.clone());
            }
            SessionEventData::AssistantMessage {
                turn: t, message, ..
            } if *t == turn => {
                let text = message.text();
                if !text.is_empty() {
                    if !reply.is_empty() {
                        reply.push('\n');
                    }
                    reply.push_str(&text);
                }
            }
            SessionEventData::TurnEnd {
                turn: t,
                reason: TurnEndReason::Error { message },
            } if *t == turn => {
                // 诚实出口：turn 失败原因浮出（工具结果回流后模型调用失败等）
                error = Some(message.clone());
            }
            _ => {}
        }
    }
    let tool_trace: Vec<String> = calls
        .into_iter()
        .map(|(call_id, display)| match results.get(&call_id) {
            Some(result) => format!("{display} → {result}"),
            None => display,
        })
        .collect();
    TurnSummary {
        turn,
        reply,
        tool_trace,
        cancelled,
        error,
    }
}

/// 收尾（三种形态共用）：不变量校验 + 会话末 digest + JSONL 落盘（含重放校验）+ 优雅卸载。
///
/// 使用装配结果里的主 agent（CLI 形态；无 LLM 时 [`Assembled::agent`] 报引导错误）。
/// 库嵌入模式（agent 由调用方自行创建）请用 [`finish_with`]。
pub async fn finish(assembled: &Assembled, config: &RunConfig) -> Result<RunReport, AppError> {
    let agent = assembled.agent()?;
    finish_with(assembled, agent, config).await
}

/// 收尾（指定 agent 版本）：库嵌入模式用——`assembled.agent` 为 None 时，
/// 把自行创建的 agent 传进来即可走同一套收尾（不变量/落盘/卸载）。
pub async fn finish_with(
    assembled: &Assembled,
    agent: &Arc<dyn Agent>,
    config: &RunConfig,
) -> Result<RunReport, AppError> {
    let Assembled { root, app, .. } = assembled;

    // 不变量：模型可见 ⟺ 已记录、seq 单调等
    let violations = root
        .get::<InvariantRegistry>()
        .expect("刚装配")
        .verify(agent.session());

    // 会话末 digest 收尾（M3，记忆插件装配时）：统计 + 转录 → 卡三段注记（慢路径）
    if let Ok(store) = root.get::<MemoryStore>() {
        let transcript = transcript_of(agent.session());
        if let Err(error) = store.digest(&transcript, cos_memory::now_ms()).await {
            eprintln!("[memory] 会话末 digest 失败: {error}");
        }
    }

    // 持久化 + 重放校验（逐事件一致）
    if let Some(path) = &config.session_path {
        // 创建时间取会话自带的：这是覆盖式重写，凭空造 0 会把日志头冲掉
        let header = SessionHeader {
            version: SESSION_FORMAT_VERSION,
            id: config.session_id.clone(),
            created_at_ms: agent.session().created_at_ms(),
            cwd: None,
        };
        save_jsonl(agent.session(), &header, path)?;
        let (loaded_header, loaded_events) = load_jsonl(path)?;
        if loaded_header.id != config.session_id || loaded_events != agent.session().events() {
            return Err(AppError::Other("会话重放不一致".into()));
        }
    }

    // 优雅退出：apply 逆序卸载（审计）
    let unload_order: Vec<String> = app
        .instances()
        .iter()
        .rev()
        .map(|instance| instance.name.clone())
        .collect();
    app.dispose_async().await;

    Ok(RunReport {
        dump: None,
        unload_order,
        events: agent.session().events(),
        messages: agent.session().derive_messages(),
        violations,
        #[cfg(feature = "plugin-todo")]
        services_after_unload: root.get::<plugin_todo::TodoStore>().is_err(),
        #[cfg(not(feature = "plugin-todo"))]
        services_after_unload: true,
    })
}

/// 一次性演示链路（`--prompt`）：装配 → 一轮 → 收尾。
pub async fn run(config: RunConfig) -> Result<RunReport, AppError> {
    if config.dump_config {
        let path = config
            .config_path
            .as_deref()
            .ok_or_else(|| AppError::Other("--dump-config 需要 cordis.yml 路径".into()))?;
        return Ok(RunReport {
            dump: Some(plugins::plan_json(path, &config.patch_files)?),
            unload_order: Vec::new(),
            events: Vec::new(),
            messages: Vec::new(),
            violations: Vec::new(),
            services_after_unload: true,
        });
    }
    let prompt = config
        .prompt
        .clone()
        .ok_or_else(|| AppError::Other("一次性模式需要 --prompt".into()))?;
    let assembled = assemble(&config).await?;
    let agent = assembled.agent()?;
    run_turn(agent, UserMessage::new(prompt), config.cancel.as_ref()).await;
    finish(&assembled, &config).await
}

/// 轮询取消信号（main 的 Ctrl-C 监视任务写入）。
pub(crate) async fn wait_for_cancel(flag: Arc<AtomicBool>) {
    loop {
        if flag.load(Ordering::Acquire) {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

/// 会话日志 → 完整转录（会话末 digest 输入；turn 边界打标）。
fn transcript_of(session: &cos_session::Session) -> String {
    let mut lines: Vec<String> = Vec::new();
    for event in session.events() {
        match &event.data {
            cos_session::SessionEventData::TurnStart { turn } => {
                lines.push(format!("— turn {turn} —"));
            }
            cos_session::SessionEventData::UserMessage(message) => {
                lines.push(format!("用户: {}", message.content));
            }
            cos_session::SessionEventData::AssistantMessage { message, .. } => {
                lines.push(format!("助手: {}", message.text()));
            }
            _ => {}
        }
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::summarize_turn;
    use cos_llm::ToolResultMessage;
    use cos_session::{Session, SessionEventData, TurnEndReason};

    /// 工具结果回流后模型调用失败 → turn/end 记 Error → 摘要必须浮出（诚实出口）。
    #[test]
    fn summarize_turn_surfaces_turn_error() {
        let session = Session::new("t");
        session.append(SessionEventData::TurnStart { turn: 1 });
        session.append(SessionEventData::ToolCall {
            turn: 1,
            step: 1,
            call_id: "c1".into(),
            name: "recall".into(),
            arguments: "{}".into(),
        });
        session.append(SessionEventData::ToolResult {
            turn: 1,
            step: 1,
            call_id: "c1".into(),
            message: ToolResultMessage {
                content: "无相关记忆".into(),
                call_id: Some("c1".into()),
            },
            error: None,
        });
        session.append(SessionEventData::TurnEnd {
            turn: 1,
            reason: TurnEndReason::Error {
                message: "HTTP 400: tool_call_id 缺失".into(),
            },
        });

        let summary = summarize_turn(&session, 1, false);
        assert!(summary.reply.is_empty());
        assert_eq!(
            summary.error.as_deref(),
            Some("HTTP 400: tool_call_id 缺失"),
            "turn 失败原因必须浮出"
        );
        assert_eq!(summary.tool_trace.len(), 1, "工具轨迹仍应显示");
    }

    /// 正常完成的 turn → error 为 None。
    #[test]
    fn summarize_turn_completed_has_no_error() {
        let session = Session::new("t");
        session.append(SessionEventData::TurnStart { turn: 1 });
        session.append(SessionEventData::TurnEnd {
            turn: 1,
            reason: TurnEndReason::Completed,
        });
        let summary = summarize_turn(&session, 1, false);
        assert!(summary.error.is_none());
    }
}
