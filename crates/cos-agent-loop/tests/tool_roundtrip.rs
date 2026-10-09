//! P5 验收：模型回 tool-call → 工具执行 → 结果回流 → 下一 step 的
//! derive_messages 含完整 call/result 对；请求带工具 schema 与装配好的 prompt。

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use cos_agent::{AgentOptions, AgentRegistry, CreateAgentOptions};
use cos_agent_loop::LoopFactory;
use cos_core::Context;
use cos_llm::{
    ChunkDelta, ContentBlock, LlmAdapter, LlmRequest, LlmStream, Message, StreamChunk, ToolCall,
    UserMessage,
};
use cos_session::{SessionEventData, TurnEndReason};
use cos_system_prompt::{PromptSection, PromptSections};
use cos_test_support::{MockAdapter, MockReply};
use cos_tools::{Tool, ToolOutcome, ToolRegistry, ToolRun};
use futures::future::BoxFuture;

/// 记录最近一次请求的适配器（透传给 mock）。
struct RecordingAdapter {
    inner: Arc<dyn LlmAdapter>,
    last: Arc<Mutex<Option<LlmRequest>>>,
}

impl LlmAdapter for RecordingAdapter {
    fn id(&self) -> &str {
        self.inner.id()
    }

    fn stream(&self, request: &LlmRequest) -> LlmStream {
        *self.last.lock().unwrap() = Some(request.clone());
        self.inner.stream(request)
    }
}

/// 记录调用次数的工具（name = "rec"）。
struct RecordingTool {
    calls: Arc<AtomicUsize>,
}

impl Tool for RecordingTool {
    fn name(&self) -> &'static str {
        "rec"
    }

    fn description(&self) -> &'static str {
        "记录工具"
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "content": { "type": "string" } },
            "required": ["content"]
        })
    }

    fn execute(
        &self,
        _ctx: &Context,
        run: &ToolRun,
    ) -> BoxFuture<'static, Result<ToolOutcome, cos_session::ToolError>> {
        let calls = self.calls.clone();
        let content = run.arguments["content"].as_str().unwrap_or("").to_string();
        Box::pin(async move {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(ToolOutcome::ok(format!("已记录: {content}")))
        })
    }
}

#[tokio::test]
async fn tool_call_executes_and_results_flow_back() {
    let root = Context::root();
    root.provide(ToolRegistry::new(&root)).unwrap();
    root.provide(PromptSections::new(&root)).unwrap();
    root.get::<PromptSections>()
        .unwrap()
        .append(PromptSection::new("persona", 10, "你是助手。"))
        .unwrap();

    let calls = Arc::new(AtomicUsize::new(0));
    root.get::<ToolRegistry>()
        .unwrap()
        .register(Arc::new(RecordingTool {
            calls: calls.clone(),
        }))
        .unwrap();

    let registry = AgentRegistry::new(&root);
    root.provide(registry.clone()).unwrap();
    registry.set_factory(Arc::new(LoopFactory)).unwrap();

    let inner: Arc<dyn LlmAdapter> = Arc::new(MockAdapter::new(
        "mock",
        vec![
            MockReply::new(vec![StreamChunk {
                delta: ChunkDelta::ToolUse {
                    call: ToolCall {
                        call_id: "c1".into(),
                        name: "rec".into(),
                        arguments: "{\"content\":\"做 P5\"}".into(),
                    },
                },
                usage: None,
            }]),
            MockReply::text("已记录"),
        ],
    ));
    let last = Arc::new(Mutex::new(None));
    let adapter: Arc<dyn LlmAdapter> = Arc::new(RecordingAdapter {
        inner,
        last: last.clone(),
    });

    let agent = registry
        .create(CreateAgentOptions {
            session: None,
            session_id: "sess-tools".into(),
            options: AgentOptions {
                provider: Some("mock".into()),
                model: Some("m".into()),
                max_tokens: None,
                role: None,
            },
            adapter,
        })
        .await
        .unwrap();

    agent.followup(UserMessage::new("帮我记录一条 todo"));
    agent.when_idle().await;

    // 日志：tool/call 先写、tool/result 后写；工具结果回流产生第二 step
    let events = agent.session().events();
    let types: Vec<&str> = events.iter().map(|event| event.data.type_name()).collect();
    let call_pos = types.iter().position(|name| *name == "tool/call").unwrap();
    let result_pos = types
        .iter()
        .position(|name| *name == "tool/result")
        .unwrap();
    assert!(call_pos < result_pos, "tool/call 必须先写日志");
    assert_eq!(
        types.iter().filter(|name| **name == "step/start").count(),
        2,
        "工具结果回流应产生第二 step"
    );

    // derive_messages：完整 call/result 对 + 回流回复
    let messages = agent.session().derive_messages();
    assert_eq!(messages.len(), 4);
    assert_eq!(
        messages[0],
        Message::User(UserMessage::new("帮我记录一条 todo"))
    );
    match &messages[1] {
        Message::Assistant(assistant) => {
            assert!(matches!(assistant.content[0], ContentBlock::ToolUse { .. }))
        }
        other => panic!("期望 Assistant(ToolUse)，实际 {other:?}"),
    }
    match &messages[2] {
        Message::Tool(tool) => assert_eq!(tool.content, "已记录: 做 P5"),
        other => panic!("期望 Tool，实际 {other:?}"),
    }
    match &messages[3] {
        Message::Assistant(assistant) => assert_eq!(assistant.text(), "已记录"),
        other => panic!("期望 Assistant，实际 {other:?}"),
    }

    // 工具体被真实调用
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // 第一步请求带工具 schema 与装配好的 system prompt
    let request = last.lock().unwrap().clone().unwrap();
    assert_eq!(request.tools.len(), 1);
    assert_eq!(request.tools[0]["function"]["name"], "rec");
    let system = request.system.as_deref().unwrap();
    assert!(system.contains("你是助手。"));
    assert!(system.contains("rec"));

    // turn 正常完成
    assert!(matches!(
        events.last().unwrap().data,
        SessionEventData::TurnEnd {
            turn: 1,
            reason: TurnEndReason::Completed
        }
    ));
}
