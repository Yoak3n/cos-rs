//! cos-agent-loop —— turn/step 驱动器，实现 cos-agent（P4，主干核心）。
//!
//! 语义权威参考：`packages/core/agent-loop/src/{agent,index,tool-calls}.ts`。
//!
//! 状态机（照 agent.ts）：`wake_driver` → `kick` → `turn`（turn/start 日志 →
//! pre-step waterfall → step：request/header 日志 → stream → assistant/chunk 逐条 →
//! assistant/message 日志 → turn/end 日志）。**每步先写日志再行动**（唯一事实源）。
//!
//! P4 边界：工具调用执行属 P5（tools 管线）——含工具块的回复按 completed 收束；
//! `agent/request-error` 重试、`agent/turn-stopping`、持久化 resume 留待后续阶段。

#![warn(missing_docs)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use cos_agent::{
    AbortSignal, Agent, AgentError, AgentErrorPayload, AgentFactory, AgentInboxPayload,
    AgentOptions, AgentStatus, AgentStatusPayload, CreateAgentOptions, Inbox, InboxTarget,
    Maintenance, PreStepDecision, PreStepPayload,
};
use cos_core::{Context, ScopeKey, ScopeTarget};
use cos_llm::{
    AssistantMessage, ChunkDelta, ContentBlock, FinishReason, LlmAdapter, LlmRequest, ToolCall,
    UserMessage,
};
use cos_session::{AbortCause, RequestHeader, Session, SessionEventData, TurnEndReason};
use futures::StreamExt;
use futures::future::BoxFuture;
use thiserror::Error;
use tokio::sync::oneshot;

/// 驱动器边界错误（内部使用；对外映射为日志与事件）。
#[derive(Debug, Error)]
enum LoopError {
    /// 被取消。
    #[error("aborted")]
    Aborted,
    /// 相位非法（私有调用在非 running 相位执行）。
    #[error("not running")]
    NotRunning,
    /// LLM 适配器失败。
    #[error("llm: {0}")]
    Llm(#[from] cos_llm::LlmError),
    /// 其他失败。
    #[error("{0}")]
    Other(String),
}

/// 运行期取消句柄（标志 + 原因）。
#[derive(Default)]
struct AbortHandle {
    aborted: AtomicBool,
    cause: Mutex<Option<AbortCause>>,
}

impl AbortHandle {
    fn abort(&self, cause: AbortCause) {
        *self.cause.lock().unwrap() = Some(cause);
        self.aborted.store(true, Ordering::Release);
    }

    fn is_aborted(&self) -> bool {
        self.aborted.load(Ordering::Acquire)
    }

    fn cause(&self) -> Option<AbortCause> {
        self.cause.lock().unwrap().clone()
    }
}

/// 驱动器相位（照 dsh `Phase`）。
enum Phase {
    /// 空闲（记录最近 turn）。
    Idle {
        /// 最近 turn 号。
        last_turn: u32,
    },
    /// 活动 turn 驱动器。
    Running {
        /// 当前 turn。
        turn: u32,
        /// 当前 step。
        step: u32,
        /// 本活动期的取消句柄。
        aborted: Arc<AbortHandle>,
    },
    /// 空闲期维护任务（公开状态仍为 idle；唤醒闩在收敛时重放）。
    Maintenance {
        /// 取消句柄。
        aborted: Arc<AbortHandle>,
        /// 任务侧取消信号。
        signal: AbortSignal,
        /// 维护期间收到的唤醒请求。
        wake_requested: bool,
        /// 最近 turn 号。
        last_turn: u32,
    },
}

/// 共享的活跃标志与空闲等待者（`when_idle` 的门闩）。
#[derive(Default)]
struct IdleGate {
    active: AtomicBool,
    waiters: Mutex<Vec<oneshot::Sender<()>>>,
}

/// 驱动器共享状态（`LoopAgent` 克隆共享同一份）。
struct AgentCore {
    id: String,
    options: AgentOptions,
    session: Session,
    inbox: Inbox,
    agent_ctx: Context,
    root: Context,
    adapter: Arc<dyn LlmAdapter>,
    phase: Arc<Mutex<Phase>>,
    gate: Arc<IdleGate>,
}

impl AgentCore {
    fn scope_target(&self) -> ScopeTarget {
        ScopeTarget::Key(ScopeKey::new(format!("agent:{}", self.id)))
    }

    fn emit<T: 'static + Send + Sync>(&self, name: &'static str, payload: T) {
        self.root
            .target(self.scope_target())
            .emit(name, Arc::new(payload));
    }

    fn emit_status(&self, status: AgentStatus) {
        self.emit(
            "agent/status",
            AgentStatusPayload {
                agent_id: self.id.clone(),
                status,
            },
        );
    }

    fn current_abort_handle(&self) -> Option<Arc<AbortHandle>> {
        match &*self.phase.lock().unwrap() {
            Phase::Running { aborted, .. } | Phase::Maintenance { aborted, .. } => {
                Some(aborted.clone())
            }
            Phase::Idle { .. } => None,
        }
    }

    fn check_abort(handle: &AbortHandle) -> Result<(), LoopError> {
        if handle.is_aborted() {
            Err(LoopError::Aborted)
        } else {
            Ok(())
        }
    }

    /// 唤醒驱动器：idle → running（+状态事件）；维护期闩唤醒；活动期自取队列。
    fn wake_driver(&self) {
        let mut phase = self.phase.lock().unwrap();
        match &mut *phase {
            Phase::Idle { last_turn } => {
                let handle = Arc::new(AbortHandle::default());
                *phase = Phase::Running {
                    turn: *last_turn,
                    step: 0,
                    aborted: handle,
                };
                drop(phase);
                self.gate.active.store(true, Ordering::SeqCst);
                self.emit_status(AgentStatus::Running);
                self.spawn_driver();
            }
            Phase::Running { .. } => { /* 活跃驱动器自己消费队列 */ }
            Phase::Maintenance { wake_requested, .. } => {
                *wake_requested = true;
            }
        }
    }

    /// 以发起者因果链派生驱动器任务。
    fn spawn_driver(&self) {
        let core = self.clone_core();
        let agent: Arc<dyn Agent> = Arc::new(LoopAgent { core: core.clone() });
        tokio::spawn(cos_agent::with_initiator(agent, async move {
            kick(core).await;
        }));
    }

    fn clone_core(&self) -> Arc<AgentCore> {
        Arc::new(AgentCore {
            id: self.id.clone(),
            options: self.options.clone(),
            session: self.session.clone(),
            inbox: self.inbox.clone(),
            agent_ctx: self.agent_ctx.clone(),
            root: self.root.clone(),
            adapter: self.adapter.clone(),
            phase: self.phase.clone(),
            gate: self.gate.clone(),
        })
    }

    /// 驱动器主体：turn 循环 + 收敛到 idle。
    async fn drive(&self) {
        while let Ok(true) = self.turn().await {}
        {
            let mut phase = self.phase.lock().unwrap();
            let last_turn = match &*phase {
                Phase::Running { turn, .. } => *turn,
                _ => 0,
            };
            *phase = Phase::Idle { last_turn };
        }
        self.gate.active.store(false, Ordering::SeqCst);
        self.emit_status(AgentStatus::Idle);
        let waiters = std::mem::take(&mut *self.gate.waiters.lock().unwrap());
        for sender in waiters {
            let _ = sender.send(());
        }
    }

    /// 一个 turn：先写 turn/start，再按 step 推进，最后写 turn/end。
    /// 返回 true = 还有待处理输入、应继续下一 turn。
    async fn turn(&self) -> Result<bool, LoopError> {
        let handle = self.current_abort_handle().ok_or(LoopError::NotRunning)?;
        let turn = {
            let mut phase = self.phase.lock().unwrap();
            match &mut *phase {
                Phase::Running { turn, .. } => {
                    *turn += 1;
                    *turn
                }
                _ => return Err(LoopError::NotRunning),
            }
        };
        // 先写日志再行动
        self.session.append(SessionEventData::TurnStart { turn });
        if let Err(error) = Self::check_abort(&handle) {
            // 调度前取消：同样闭合 turn（start↔end 配对不变量，先写日志再返回）
            self.session.append(SessionEventData::TurnEnd {
                turn,
                reason: TurnEndReason::Aborted {
                    cause: handle.cause().unwrap_or(AbortCause::User),
                },
            });
            return Err(error);
        }

        let mut turn_ends: Option<TurnEndReason> = None;
        let mut target = InboxTarget::NextTurn;
        let result: Result<(), LoopError> = async {
            loop {
                Self::check_abort(&handle)?;
                let step = {
                    let mut phase = self.phase.lock().unwrap();
                    match &mut *phase {
                        Phase::Running { step, .. } => {
                            *step += 1;
                            *step
                        }
                        _ => return Err(LoopError::NotRunning),
                    }
                };
                let claimed = self.inbox.claim(target);
                // pre-step waterfall：默认 enter（带原消息）；监听器可 Reject / 替换消息
                let payload = PreStepPayload {
                    agent_id: self.id.clone(),
                    messages: claimed,
                    turn,
                    step,
                };
                let decision = self
                    .root
                    .target(self.scope_target())
                    .waterfall("agent/pre-step", payload.clone(), |d| {
                        Box::pin(async move {
                            PreStepDecision::Enter {
                                messages: d.value().messages.clone(),
                            }
                        })
                    })
                    .await
                    .map_err(|error| LoopError::Other(error.to_string()))?;
                Self::check_abort(&handle)?;

                let messages = match decision {
                    PreStepDecision::Reject => {
                        turn_ends = Some(TurnEndReason::Blocked);
                        return Ok(());
                    }
                    PreStepDecision::Enter { messages } => messages,
                };
                if messages.is_empty() {
                    // 空进入：首 step 无消息 → completed；turn 未结束且无新消息 →
                    // 工具结果回流 step（继续）；否则收敛（不花模型调用）
                    if step == 1 {
                        turn_ends = Some(TurnEndReason::Completed);
                        return Ok(());
                    }
                    if turn_ends.is_some() {
                        break;
                    }
                }

                // 先写日志再行动：step/start + user/message
                self.session
                    .append(SessionEventData::StepStart { turn, step });
                for message in &messages {
                    self.session
                        .append(SessionEventData::UserMessage(message.clone()));
                }
                match self.run_step(turn, step, &handle).await {
                    Ok(Some(reason)) => turn_ends = Some(reason),
                    Ok(None) => { /* 工具已执行，turn 继续（结果回流下一 step） */ }
                    Err(LoopError::Aborted) => {
                        turn_ends = Some(TurnEndReason::Aborted {
                            cause: handle.cause().unwrap_or(AbortCause::User),
                        });
                        // 先写日志：失败路径同样闭合 step（step/start 与 step/end 配对）
                        self.session
                            .append(SessionEventData::StepEnd { turn, step });
                        return Err(LoopError::Aborted);
                    }
                    Err(error) => {
                        turn_ends = Some(TurnEndReason::Error {
                            message: error.to_string(),
                        });
                        self.emit(
                            "agent/error",
                            AgentErrorPayload {
                                agent_id: self.id.clone(),
                                turn,
                                step,
                                message: error.to_string(),
                            },
                        );
                        // 先写日志：失败路径同样闭合 step（step/start 与 step/end 配对）
                        self.session
                            .append(SessionEventData::StepEnd { turn, step });
                        return Err(error);
                    }
                }
                self.session
                    .append(SessionEventData::StepEnd { turn, step });

                if turn_ends.is_some() && self.inbox.next_step_len() == 0 {
                    break;
                }
                target = InboxTarget::NextStep;
            }
            Ok(())
        }
        .await;

        // 无论成败，写 turn/end（先写日志）
        let reason = turn_ends.clone().unwrap_or_else(|| match &result {
            Err(LoopError::Aborted) => TurnEndReason::Aborted {
                cause: handle.cause().unwrap_or(AbortCause::User),
            },
            Err(error) => TurnEndReason::Error {
                message: error.to_string(),
            },
            Ok(()) => TurnEndReason::Completed,
        });
        self.session
            .append(SessionEventData::TurnEnd { turn, reason });
        result?;

        if self.inbox.has_pending() {
            Self::check_abort(&handle)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// 一个 step：build_request（agent/request waterfall）→ request/header 日志 →
    /// stream → assistant/chunk 逐条 → assistant/message 日志 →
    /// 工具执行（tool/call 先写日志 → 管线 → tool/result 日志）。
    ///
    /// 返回 `Some(原因)` = turn 收束；`None` = 已执行工具、turn 继续（结果回流下一 step）。
    async fn run_step(
        &self,
        turn: u32,
        step: u32,
        handle: &AbortHandle,
    ) -> Result<Option<TurnEndReason>, LoopError> {
        Self::check_abort(handle)?;
        // 工具 schema 与 system prompt 来自接缝服务（缺省 = 无工具/无 system）
        let tool_registry = self.agent_ctx.get::<cos_tools::ToolRegistry>().ok();
        let tools: Vec<serde_json::Value> = tool_registry
            .as_ref()
            .map(|registry| registry.schemas())
            .unwrap_or_default();
        let system = self
            .agent_ctx
            .get::<cos_system_prompt::PromptSections>()
            .ok()
            .map(|sections| sections.render());
        let messages = self.session.derive_messages();
        let request = LlmRequest {
            system,
            messages,
            tools: tools.clone(),
        };
        // agent/request waterfall：默认 = 当前请求；监听器可替换
        let request = self
            .root
            .target(self.scope_target())
            .waterfall("agent/request", request.clone(), |d| {
                Box::pin(async move { d.value().clone() })
            })
            .await
            .map_err(|error| LoopError::Other(error.to_string()))?;
        Self::check_abort(handle)?;

        // 先写日志再行动：request/header
        self.session.append(SessionEventData::RequestHeader {
            header: RequestHeader {
                config: serde_json::json!({
                    "provider": self.options.provider,
                    "model": self.options.model,
                    "maxTokens": self.options.max_tokens,
                }),
                system: request.system.clone(),
                tools: if request.tools.is_empty() {
                    None
                } else {
                    Some(serde_json::json!(request.tools))
                },
            },
        });

        let mut blocks: Vec<ContentBlock> = Vec::new();
        let mut usage = None;
        let mut finish_reason: Option<FinishReason> = None;
        let mut stream = self.adapter.stream(&request);
        while let Some(item) = stream.next().await {
            Self::check_abort(handle)?;
            match item {
                Ok(chunk) => {
                    // 终结分片 = 控制信息（非内容）：记录结束原因，不入会话日志
                    if let ChunkDelta::Finish { reason } = chunk.delta {
                        finish_reason = Some(reason);
                        continue;
                    }
                    // 先写日志：chunk 逐条入账
                    self.session.append(SessionEventData::AssistantChunk {
                        turn,
                        step,
                        chunk: chunk.clone(),
                    });
                    if chunk.usage.is_some() {
                        usage = chunk.usage;
                    }
                    match chunk.delta {
                        ChunkDelta::Text { text } => {
                            if text.is_empty() {
                                continue;
                            }
                            // 相邻文本块合并（同 dsh BlockAssembler）
                            match blocks.last_mut() {
                                Some(ContentBlock::Text { text: tail }) => tail.push_str(&text),
                                _ => blocks.push(ContentBlock::Text { text }),
                            }
                        }
                        ChunkDelta::Thinking { text } => {
                            if text.is_empty() {
                                continue;
                            }
                            // 推理与正文分开成块（消费方选择性展示）
                            match blocks.last_mut() {
                                Some(ContentBlock::Thinking { text: tail }) => tail.push_str(&text),
                                _ => blocks.push(ContentBlock::Thinking { text }),
                            }
                        }
                        ChunkDelta::ToolUse { call } => blocks.push(ContentBlock::ToolUse { call }),
                        ChunkDelta::Finish { .. } => unreachable!("已在上方拦截"),
                    }
                }
                Err(error) => return Err(LoopError::Llm(error)),
            }
        }
        Self::check_abort(handle)?;

        // 先写日志：assistant/message（推导历史用这条）
        let tool_calls: Vec<ToolCall> = blocks
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolUse { call } => Some(call.clone()),
                ContentBlock::Text { .. } | ContentBlock::Thinking { .. } => None,
            })
            .collect();
        self.session.append(SessionEventData::AssistantMessage {
            turn,
            step,
            message: AssistantMessage::new(blocks),
            usage,
        });

        if tool_calls.is_empty() && finish_reason != Some(FinishReason::ToolCalls) {
            return Ok(Some(TurnEndReason::Completed));
        }
        if tool_calls.is_empty() {
            // 模型显式请求工具（finish: tool-calls）但未产出任何工具块（适配器异常）：
            // 按完成处理，避免死循环（不执行空工具轮）
            return Ok(Some(TurnEndReason::Completed));
        }

        // 先写日志再行动：tool/call ×N（模型序、顺序执行）
        for call in &tool_calls {
            self.session.append(SessionEventData::ToolCall {
                turn,
                step,
                call_id: call.call_id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            });
        }
        // 执行管线（tools 服务缺省 → 错误结果）
        for call in &tool_calls {
            Self::check_abort(handle)?;
            let run = cos_tools::ToolRun {
                call_id: call.call_id.clone(),
                name: call.name.clone(),
                arguments: parse_arguments(&call.arguments),
                turn,
                step,
                caller: cos_tools::ToolCaller {
                    role: self
                        .options
                        .role
                        .clone()
                        .unwrap_or_else(|| "agent".to_string()),
                },
            };
            let outcome = match &tool_registry {
                Some(registry) => registry.execute(&self.agent_ctx, &run).await,
                None => cos_tools::ToolOutcome::error(
                    "tools 服务未注册".to_string(),
                    cos_session::ToolError {
                        name: "ToolRegistry".into(),
                        code: "NO_TOOLS".into(),
                    },
                ),
            };
            // 写日志：tool/result（先写日志）
            self.session.append(SessionEventData::ToolResult {
                turn,
                step,
                call_id: call.call_id.clone(),
                message: cos_llm::ToolResultMessage {
                    content: outcome.content.clone(),
                    call_id: Some(call.call_id.clone()),
                },
                error: outcome.error.clone(),
            });
            // 实时通知（冻结结果之后）
            self.emit(
                "tools/result",
                cos_tools::ToolResultPayload {
                    call_id: call.call_id.clone(),
                    name: call.name.clone(),
                    content: outcome.content,
                    is_error: outcome.is_error,
                },
            );
        }
        Self::check_abort(handle)?;
        Ok(None)
    }
}

/// 驱动器任务入口：循环 + 收敛。
async fn kick(core: Arc<AgentCore>) {
    core.drive().await;
}

/// 解析模型产出的参数（同 dsh `parseArguments`）：空串 → `{}`，非法 JSON → 原串文本。
fn parse_arguments(raw: &str) -> serde_json::Value {
    if raw.trim().is_empty() {
        return serde_json::json!({});
    }
    serde_json::from_str(raw).unwrap_or(serde_json::Value::String(raw.to_string()))
}

/// 等待活跃标志清除（`when_idle` 的实现）。
async fn wait_idle(gate: Arc<IdleGate>) {
    loop {
        if !gate.active.load(Ordering::Acquire) {
            return;
        }
        let (sender, receiver) = oneshot::channel();
        {
            let mut waiters = gate.waiters.lock().unwrap();
            if !gate.active.load(Ordering::Acquire) {
                return;
            }
            waiters.push(sender);
        }
        let _ = receiver.await;
    }
}

/// 默认 turn/step 驱动器（实现 [`Agent`]）。
#[derive(Clone)]
pub struct LoopAgent {
    core: Arc<AgentCore>,
}

impl Agent for LoopAgent {
    fn id(&self) -> &str {
        &self.core.id
    }

    fn options(&self) -> &AgentOptions {
        &self.core.options
    }

    fn session(&self) -> &Session {
        &self.core.session
    }

    fn ctx(&self) -> &Context {
        &self.core.agent_ctx
    }

    fn status(&self) -> AgentStatus {
        match &*self.core.phase.lock().unwrap() {
            Phase::Idle { .. } | Phase::Maintenance { .. } => AgentStatus::Idle,
            Phase::Running { .. } => AgentStatus::Running,
        }
    }

    fn send(&self, message: UserMessage, target: InboxTarget, wake: bool) {
        self.core.inbox.push(target, message.clone());
        self.core.emit(
            "agent/inbox/inserted",
            AgentInboxPayload {
                agent_id: self.core.id.clone(),
                message,
            },
        );
        if wake {
            self.core.wake_driver();
        }
    }

    fn followup(&self, message: UserMessage) {
        self.send(message, InboxTarget::NextTurn, true);
    }

    fn steer(&self, message: UserMessage) {
        self.send(message, InboxTarget::NextStep, true);
    }

    fn inject(&self, message: UserMessage) {
        self.send(message, InboxTarget::NextStep, false);
    }

    fn pending_count(&self) -> usize {
        self.core.inbox.next_turn_len() + self.core.inbox.next_step_len()
    }

    fn cancel_message(&self, id: &str) -> bool {
        self.core.inbox.remove_by_id(id)
    }

    fn has_queued_message(&self, id: &str) -> bool {
        self.core.inbox.contains_id(id)
    }

    fn cancel(&self, cause: AbortCause, keep_inbox: bool) {
        if !keep_inbox {
            self.core.inbox.clear();
        }
        let mut phase = self.core.phase.lock().unwrap();
        match &mut *phase {
            Phase::Running { aborted, .. } => aborted.abort(cause),
            Phase::Maintenance {
                aborted, signal, ..
            } => {
                aborted.abort(cause);
                signal.abort();
            }
            Phase::Idle { .. } => {}
        }
    }

    fn when_idle(&self) -> BoxFuture<'static, ()> {
        let gate = self.core.gate.clone();
        Box::pin(wait_idle(gate))
    }

    fn run_maintenance(&self, task: Maintenance) -> BoxFuture<'static, Result<(), AgentError>> {
        let core = self.core.clone_core();
        Box::pin(async move {
            let signal = {
                let mut phase = core.phase.lock().unwrap();
                match &*phase {
                    Phase::Idle { last_turn } => {
                        let signal = AbortSignal::new();
                        *phase = Phase::Maintenance {
                            aborted: Arc::new(AbortHandle::default()),
                            signal: signal.clone(),
                            wake_requested: false,
                            last_turn: *last_turn,
                        };
                        signal
                    }
                    _ => return Err(AgentError::Busy(core.id.clone())),
                }
            };
            core.gate.active.store(true, Ordering::SeqCst);
            task(signal).await;
            // 收敛：维护期闩的唤醒在 idle 后重放
            let wake_requested = {
                let mut phase = core.phase.lock().unwrap();
                let (last_turn, wake_requested) = match &*phase {
                    Phase::Maintenance {
                        last_turn,
                        wake_requested,
                        ..
                    } => (*last_turn, *wake_requested),
                    _ => (0, false),
                };
                *phase = Phase::Idle { last_turn };
                wake_requested
            };
            core.gate.active.store(false, Ordering::SeqCst);
            let waiters = std::mem::take(&mut *core.gate.waiters.lock().unwrap());
            for sender in waiters {
                let _ = sender.send(());
            }
            if wake_requested && core.inbox.has_pending() {
                core.wake_driver();
            }
            Ok(())
        })
    }
}

/// 默认工厂：`ctx.agents` 注册的创建入口。
#[derive(Default)]
pub struct LoopFactory;

/// 默认 agent 驱动 id（`--agent-driver <id>` 选择键；宿主锚点用）。
pub const LOOP_DRIVER_ID: &str = "loop";

/// 驱动工厂构建函数（`agent_factory!` 注册）：配置忽略 → [`LoopFactory`]。
pub fn build_loop(_config: &serde_json::Value) -> Result<Arc<dyn AgentFactory>, AgentError> {
    Ok(Arc::new(LoopFactory))
}

cos_agent::agent_factory!("loop", build_loop);

impl AgentFactory for LoopFactory {
    fn create(
        &self,
        root: &Context,
        options: CreateAgentOptions,
    ) -> BoxFuture<'static, Result<Arc<dyn Agent>, AgentError>> {
        let root = root.clone();
        Box::pin(async move {
            let agent_ctx =
                root.fork_scoped(ScopeKey::new(format!("agent:{}", options.session_id)));
            let session = options
                .session
                .clone()
                .unwrap_or_else(|| Session::new(options.session_id.clone()));
            // turn 号从会话日志续接：驱动器实例的生命周期不等于会话的生命周期。
            let last_turn = session.last_turn();
            let core = Arc::new(AgentCore {
                id: options.session_id.clone(),
                options: options.options.clone(),
                session,
                inbox: Inbox::new(),
                agent_ctx,
                root,
                adapter: options.adapter.clone(),
                phase: Arc::new(Mutex::new(Phase::Idle { last_turn })),
                gate: Arc::new(IdleGate::default()),
            });
            Ok(Arc::new(LoopAgent { core }) as Arc<dyn Agent>)
        })
    }
}
