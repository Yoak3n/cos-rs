//! `RetryAdapter` 的触发条件与边界：首片未出才重试、可重试错误才重试、
//! 出片后不重来、次数用尽后原样交付错误。

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cos_llm::{
    ChunkDelta, LlmAdapter, LlmError, LlmErrorCode, LlmRequest, LlmStream, Message, ProviderFacts,
    RetryAdapter, RetryPolicy, StreamChunk,
};
use futures::StreamExt;

fn text_chunk(text: &str) -> StreamChunk {
    StreamChunk {
        delta: ChunkDelta::Text {
            text: text.to_string(),
        },
        usage: None,
    }
}

/// 脚本化适配器：按脚本依次交付「失败 / 一堆分片 / 先分片后失败」。
enum Step {
    Fails(LlmError),
    Yields(Vec<StreamChunk>),
    YieldsThenFails(Vec<StreamChunk>, LlmError),
}

struct ScriptAdapter {
    steps: Mutex<VecDeque<Step>>,
    calls: AtomicUsize,
}

impl ScriptAdapter {
    fn new(steps: Vec<Step>) -> Arc<Self> {
        Arc::new(Self {
            steps: Mutex::new(steps.into()),
            calls: AtomicUsize::new(0),
        })
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl LlmAdapter for ScriptAdapter {
    fn id(&self) -> &str {
        "script"
    }

    fn stream(&self, _request: &LlmRequest) -> LlmStream {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let step = self
            .steps
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or(Step::Yields(Vec::new()));
        match step {
            Step::Fails(error) => Box::pin(futures::stream::once(async move { Err(error) })),
            Step::Yields(chunks) => Box::pin(futures::stream::iter(chunks.into_iter().map(Ok))),
            Step::YieldsThenFails(chunks, error) => {
                let head = futures::stream::iter(chunks.into_iter().map(Ok));
                Box::pin(head.chain(futures::stream::once(async move { Err(error) })))
            }
        }
    }
}

fn request() -> LlmRequest {
    LlmRequest {
        system: None,
        messages: vec![Message::User(cos_llm::UserMessage::new("hi"))],
        tools: Vec::new(),
    }
}

fn fast() -> RetryPolicy {
    RetryPolicy {
        max_attempts: 3,
        base_delay: Duration::ZERO,
        max_delay: Duration::ZERO,
    }
}

#[tokio::test]
async fn retries_network_errors_before_the_first_chunk() {
    let adapter = ScriptAdapter::new(vec![
        Step::Fails(LlmError::new(LlmErrorCode::Network, "连接断了")),
        Step::Fails(LlmError::new(LlmErrorCode::Network, "还没好")),
        Step::Yields(vec![text_chunk("正文")]),
    ]);
    let retry = RetryAdapter::new(adapter.clone(), fast());
    let chunks: Vec<_> = retry.stream(&request()).collect().await;
    assert_eq!(adapter.calls(), 3, "两次失败后应第三次成功");
    assert_eq!(chunks.len(), 1, "只把成功那次的正文交出去");
    assert!(
        matches!(&chunks[0], Ok(StreamChunk { delta: ChunkDelta::Text { text }, .. }) if text == "正文")
    );
}

#[tokio::test]
async fn does_not_retry_after_a_chunk_was_delivered() {
    let adapter = ScriptAdapter::new(vec![Step::YieldsThenFails(
        vec![text_chunk("已经上屏")],
        LlmError::new(LlmErrorCode::Network, "中途断"),
    )]);
    let retry = RetryAdapter::new(adapter.clone(), fast());
    let chunks: Vec<_> = retry.stream(&request()).collect().await;
    assert_eq!(adapter.calls(), 1, "已经出片就不能重来（会重复内容）");
    assert!(chunks[0].is_ok());
    assert!(chunks.last().unwrap().is_err());
}

#[tokio::test]
async fn does_not_retry_non_retryable_errors() {
    let adapter = ScriptAdapter::new(vec![Step::Fails(LlmError::new(LlmErrorCode::Auth, "401"))]);
    let retry = RetryAdapter::new(adapter.clone(), fast());
    let chunks: Vec<_> = retry.stream(&request()).collect().await;
    assert_eq!(adapter.calls(), 1, "鉴权失败重试不会改善");
    assert!(matches!(&chunks[0], Err(error) if error.code == LlmErrorCode::Auth));
}

#[tokio::test]
async fn gives_up_after_max_attempts() {
    let adapter = ScriptAdapter::new(vec![
        Step::Fails(LlmError::new(LlmErrorCode::Network, "断")),
        Step::Fails(LlmError::new(LlmErrorCode::Network, "还断")),
        Step::Fails(LlmError::new(LlmErrorCode::Network, "仍然断")),
    ]);
    let retry = RetryAdapter::new(adapter.clone(), fast());
    let chunks: Vec<_> = retry.stream(&request()).collect().await;
    assert_eq!(adapter.calls(), 3);
    assert_eq!(chunks.len(), 1);
    let error = chunks[0]
        .as_ref()
        .expect_err("最后一次失败要原样交给调用方");
    assert!(
        error.message.contains("已退避重试 2 次"),
        "错误文本要能看出重试跑过: {}",
        error.message
    );
}

#[test]
fn delay_grows_exponentially_and_honours_retry_after() {
    let policy = RetryPolicy::default();
    let plain = LlmError::new(LlmErrorCode::Network, "断");
    assert_eq!(policy.delay(1, &plain), Duration::from_secs(1));
    assert_eq!(policy.delay(2, &plain), Duration::from_secs(2));
    assert_eq!(policy.delay(3, &plain), Duration::from_secs(4));
    assert_eq!(policy.delay(10, &plain), policy.max_delay);

    let mut throttled = LlmError::new(LlmErrorCode::RateLimit, "429");
    throttled.facts = Some(ProviderFacts {
        status: Some(429),
        retry_after_ms: Some(3_000),
    });
    assert_eq!(policy.delay(1, &throttled), Duration::from_secs(3));
}
