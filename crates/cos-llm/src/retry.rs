//! 请求层**指数退避**：首片未出前失败，等一会儿再打一次。
//!
//! 为什么在适配器外面再包一层：内置适配器内部只有「流式失败 → 立刻非流式」的
//! 一次兜底（见 `openai.rs::run_request`），没有等待；偶发断网 / 代理抖动时两次
//! 请求落在同一个故障窗口里，整轮直接失败。退避必须发生在**请求层**——放到 turn
//! 层重试会把已经跑过的工具再跑一遍。
//!
//! 边界：
//! - **只在首片未出时重试**。已经流出正文再重来会重复内容（消费方已经上屏/入账）；
//! - 只重试 [`LlmError::is_retryable`] 的错误（服务端 / 限流 / 网络 / 未分类）；
//!   鉴权、配额、参数、协议错误重试不会改善；
//! - 空流也按可重试处理（网关偶发只回 usage、不送内容分片）；
//! - [`ProviderFacts::retry_after_ms`]（429 等）优先，但不突破 `max_delay`。
//!
//! 与 [`FallbackAdapter`](crate::FallbackAdapter) 的组合由调用方决定顺序：
//! `RetryAdapter::new(FallbackAdapter, …)` = 整条链失败后等待再试整条链；
//! `FallbackAdapter::new(vec![RetryAdapter(主), RetryAdapter(备)])` = 每个提供商
//! 各自重试后再切换。两种语义都成立，别同时套两层（attempts 会相乘）。

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;

use crate::{InputContent, LlmAdapter, LlmError, LlmErrorCode, LlmRequest, LlmStream};

/// 重试策略（次数含首次；延迟按 `base * 2^(n-1)` 递增，封顶 `max_delay`）。
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    /// 总尝试次数（含首次）。
    pub max_attempts: u32,
    /// 第一次重试前的等待。
    pub base_delay: Duration,
    /// 单次等待上限。
    pub max_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        // 1s / 2s / 4s：瞬时断网与代理重连通常在这个窗口内恢复；
        // 再长会让「确实连不上」的等待变得难以忍受。
        Self {
            max_attempts: 4,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(16),
        }
    }
}

impl RetryPolicy {
    /// 第 `attempt` 次失败后要等多久（`attempt` 从 1 起：1s、2s、4s…）。
    pub fn delay(&self, attempt: u32, error: &LlmError) -> Duration {
        let exponent = attempt.saturating_sub(1).min(10);
        let backoff = self
            .base_delay
            .saturating_mul(2u32.saturating_pow(exponent))
            .min(self.max_delay);
        // 提供方明确给了重试延迟就听它的，但不超过上限
        match error.facts.as_ref().and_then(|facts| facts.retry_after_ms) {
            Some(millis) => backoff.max(Duration::from_millis(millis).min(self.max_delay)),
            None => backoff,
        }
    }
}

/// 带指数退避的适配器包装。
pub struct RetryAdapter {
    inner: Arc<dyn LlmAdapter>,
    policy: RetryPolicy,
}

impl RetryAdapter {
    /// 包装一个适配器。
    pub fn new(inner: Arc<dyn LlmAdapter>, policy: RetryPolicy) -> Self {
        Self { inner, policy }
    }

    /// 当前策略。
    pub fn policy(&self) -> RetryPolicy {
        self.policy
    }
}

impl LlmAdapter for RetryAdapter {
    fn id(&self) -> &str {
        self.inner.id()
    }

    fn input_content(&self) -> &[InputContent] {
        self.inner.input_content()
    }

    fn stream(&self, request: &LlmRequest) -> LlmStream {
        let inner = self.inner.clone();
        let request = request.clone();
        let policy = self.policy;
        Box::pin(futures::stream::unfold(
            (Phase::Fresh { attempt: 1 }, inner, request, policy),
            |(phase, adapter, request, policy)| async move {
                let mut phase = phase;
                loop {
                    match phase {
                        Phase::Fresh { attempt } => {
                            phase = Phase::Active {
                                attempt,
                                sent: 0,
                                stream: adapter.stream(&request),
                            };
                        }
                        Phase::Active {
                            attempt,
                            sent,
                            mut stream,
                        } => match stream.next().await {
                            Some(Ok(chunk)) => {
                                // 已经出片：之后再失败也不重试（避免重复内容）
                                let keep = Phase::Active {
                                    attempt,
                                    sent: sent + 1,
                                    stream,
                                };
                                return Some((Ok(chunk), (keep, adapter, request, policy)));
                            }
                            Some(Err(error)) => {
                                if sent == 0
                                    && attempt < policy.max_attempts
                                    && error.is_retryable()
                                {
                                    phase = Phase::Wait {
                                        attempt: attempt + 1,
                                        delay: policy.delay(attempt, &error),
                                    };
                                } else {
                                    return Some((
                                        Err(retried_error(error, attempt)),
                                        (Phase::Done, adapter, request, policy),
                                    ));
                                }
                            }
                            None if sent == 0 => {
                                let error = LlmError::new(
                                    LlmErrorCode::Other,
                                    "空响应流（未产出任何内容）",
                                );
                                if attempt < policy.max_attempts {
                                    phase = Phase::Wait {
                                        attempt: attempt + 1,
                                        delay: policy.delay(attempt, &error),
                                    };
                                } else {
                                    return Some((
                                        Err(retried_error(error, attempt)),
                                        (Phase::Done, adapter, request, policy),
                                    ));
                                }
                            }
                            None => return None,
                        },
                        Phase::Wait { attempt, delay } => {
                            tokio::time::sleep(delay).await;
                            phase = Phase::Fresh { attempt };
                        }
                        Phase::Done => return None,
                    }
                }
            },
        ))
    }
}

/// 最终失败时把「退避重试过几次」写进错误文本（调用方能看到重试确实跑过）。
fn retried_error(mut error: LlmError, attempt: u32) -> LlmError {
    if attempt > 1 {
        error.message = format!("{}（已退避重试 {} 次）", error.message, attempt - 1);
    }
    error
}

/// 一次请求的状态机（`Fresh → Active → Wait → …`；`Done` = 错误已交付）。
enum Phase {
    /// 还没发出这一次请求。
    Fresh { attempt: u32 },
    /// 正在读这一次请求的流。
    Active {
        attempt: u32,
        /// 这次请求已交付的分片数（> 0 后不再重试）。
        sent: usize,
        stream: LlmStream,
    },
    /// 退避等待后重发。
    Wait { attempt: u32, delay: Duration },
    /// 终态：错误已经交付，下一次 poll 结束流。
    Done,
}
