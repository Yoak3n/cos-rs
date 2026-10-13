//! OpenAI 兼容 `chat/completions` 适配器（api style: `"openai"`；`cos-llm` 的
//! **`adapters` feature**）。
//!
//! `stream()` 是同步方法（[`LlmAdapter`] 对象安全接缝），故内部 `tokio::spawn` 转发：
//! 请求在后台任务执行，chunk 经 unbounded channel 流入返回的流；调用方必须在
//! tokio runtime 内。
//!
//! 协议：`POST {base_url}/chat/completions`，优先 `stream: true`（SSE 逐行
//! `data: {…}`，`choices[0].delta.content` 累积为文本块，usage 块映射 [`TokenUsage`]，
//! `data: [DONE]` 收束）；若流式在**未产出任何 chunk 前**失败于服务端（HTTP 5xx /
//! `{"type":"error",…}` 块），自动退化为非流式单次请求（`choices[0].message.content` +
//! usage 合成一个 chunk）——兼容流式暂不可用的网关（实测 opencode zen/go 流式不稳定）。
//! 4xx（鉴权/余额）不重试，原样报错。工具调用：流式按 `index` 累积 `delta.tool_calls`
//! 分片、流尾合成 [`ToolCall`]；非流式直接解析 `message.tool_calls`。推理内容
//! （`reasoning_content`）作为独立的 Thinking 增量流式转发，与正文（Text）分开，
//! 由消费方决定是否展示——不再混入文本、也不再丢弃。
//!
//! 插件化装配：本模块是 Provider **实现**（[`build_openai`]），**不含注册**——kind 由
//! 封装插件声明（plugin-opencode-provider 的 `kind: opencode`、plugin-deepseek-provider
//! 的 `kind: deepseek`、plugin-custom-provider 的 `kind: custom` 均经
//! [`crate::build_with_style`] 按 `api_style` 分发到本模块/其他风格），在 apply 时经
//! [`LlmRegistry::register_factory`] 声明式注册。**依赖纪律**：本模块随 `adapters`
//! feature 引入 reqwest/tokio——只有 Provider 封装插件开启该 feature，普通消费者
//! （memory/agent 等）用默认特性即零网络依赖。
//!
//! 历史：原为独立 crate `cos-llm-openai`（P9 决策：并入 cos-llm 的 feature 门控模块）。

use std::sync::Arc;

use crate::{
    ChunkDelta, ContentBlock, FinishReason, InputContent, LlmAdapter, LlmError, LlmErrorCode,
    LlmRequest, LlmStream, Message, StreamChunk, TokenUsage, ToolCall, truncate,
};
use futures::StreamExt;
use serde::Deserialize;

/// 提供商工厂构建函数（`llm_factory!` 注册）：配置 `{base_url, api_key, model, streaming?, max_tokens?}`。
///
/// 通用 OpenAI 兼容构建器——封装插件（opencode/deepseek/custom）注册为各自的 kind。
pub fn build_openai(config: &serde_json::Value) -> Result<Arc<dyn LlmAdapter>, LlmError> {
    /// 提供商配置（插件 yml 里的 `config` 段）。
    #[derive(Deserialize)]
    struct ProviderConfig {
        base_url: String,
        api_key: String,
        model: String,
        #[serde(default = "default_streaming")]
        streaming: bool,
        /// 输出预算（缺省 4096；推理模型无预算会把输出全花在 reasoning 上、content 出不来）。
        #[serde(default = "default_max_tokens")]
        max_tokens: Option<u32>,
        /// 可输入内容标注（缺省 text；视觉模型声明 image）。
        #[serde(default)]
        input_content: Vec<InputContent>,
        /// 额外请求头（键值映射；按键名排序后发送，保证确定性）。
        #[serde(default)]
        extra_headers: std::collections::HashMap<String, String>,
    }
    fn default_streaming() -> bool {
        // opencode zen/go 网关流式不稳定（500 / 只出推理文本），默认非流式最稳
        false
    }
    fn default_max_tokens() -> Option<u32> {
        // 推理模型思考会吃掉预算：2048 太小（思考 + 正文可能截断成空 content），给到 4096
        Some(4096)
    }
    let config: ProviderConfig = serde_json::from_value(config.clone()).map_err(|error| {
        LlmError::new(
            LlmErrorCode::InvalidRequest,
            format!("OpenAI 兼容提供商配置无效: {error}"),
        )
    })?;
    let input_content = if config.input_content.is_empty() {
        vec![InputContent::Text]
    } else {
        config.input_content
    };
    let mut extra_headers: Vec<(String, String)> = config.extra_headers.into_iter().collect();
    extra_headers.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(Arc::new(OpenAiAdapter::new(OpenAiConfig {
        base_url: config.base_url,
        api_key: config.api_key,
        model: config.model,
        streaming: config.streaming,
        max_tokens: config.max_tokens,
        input_content,
        extra_headers,
    })))
}

/// 适配器配置。
#[derive(Debug, Clone)]
pub struct OpenAiConfig {
    /// base URL（不带 `/chat/completions` 后缀，如 `https://api.deepseek.com`）。
    pub base_url: String,
    /// API key。
    pub api_key: String,
    /// 模型 id。
    pub model: String,
    /// 是否用流式（`stream:true`）；false = 直接非流式单次请求。
    /// 某些网关（opencode zen/go）流式只出 `reasoning_content` 且时有 500，
    /// 非流式反而给出完整 `content` —— 此时关掉流式更稳。
    pub streaming: bool,
    /// 输出预算（None = 不发送）；推理模型无预算会把输出全花在 reasoning 上。
    pub max_tokens: Option<u32>,
    /// 可输入内容标注（`input_content()` 依据；视觉模型含 [`InputContent::Image`]）。
    pub input_content: Vec<InputContent>,
    /// 额外请求头（键值对，按序追加在鉴权头之后）。
    /// 网关要求会话标识（如 opencode go 的 `x-opencode-session`）或自报 UA 时用；
    /// 发往不认识这些头的端点无害（会被忽略）。
    pub extra_headers: Vec<(String, String)>,
}

/// OpenAI 兼容适配器（流式优先、非流式自动兜底）。
#[derive(Clone)]
pub struct OpenAiAdapter {
    config: OpenAiConfig,
    client: reqwest::Client,
}

/// SSE `data:` 行结构（OpenAI chat.completion.chunk）。
#[derive(Deserialize)]
struct SseChunk {
    #[serde(default)]
    choices: Vec<SseChoice>,
    #[serde(default)]
    usage: Option<SseUsage>,
}

/// SSE choices 元素。
#[derive(Deserialize)]
struct SseChoice {
    #[serde(default)]
    delta: SseDelta,
}

/// SSE delta。
#[derive(Deserialize, Default)]
struct SseDelta {
    #[serde(default)]
    content: Option<String>,
    /// 推理模型（deepseek 等）把思考文本放这里；content 缺失时兜底。
    #[serde(default)]
    reasoning_content: Option<String>,
    /// 工具调用增量（流式按 `index` 分片到达，`arguments` 为片段）。
    #[serde(default)]
    tool_calls: Vec<SseToolCall>,
}

/// OpenAI 工具调用（流式与非流式共用；流式时 `index` 用于分片归组）。
#[derive(Deserialize, Default)]
struct SseToolCall {
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: SseToolFunction,
}

/// 工具函数（名称 + 参数字符串）。
#[derive(Deserialize, Default)]
struct SseToolFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

/// 非流式响应（OpenAI chat.completion）。
#[derive(Deserialize)]
struct Completion {
    #[serde(default)]
    choices: Vec<CompletionChoice>,
    #[serde(default)]
    usage: Option<SseUsage>,
}

/// 非流式 choices 元素。
#[derive(Deserialize)]
struct CompletionChoice {
    message: CompletionMessage,
}

/// 非流式 message（content 为 null 时兜底 reasoning_content；工具调用独立解析）。
#[derive(Deserialize)]
struct CompletionMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<SseToolCall>,
}

/// usage 块（流式与非流式共用）。
#[derive(Deserialize)]
struct SseUsage {
    /// 输入 token。
    #[serde(rename = "prompt_tokens")]
    prompt_tokens: u64,
    /// 输出 token。
    #[serde(rename = "completion_tokens")]
    completion_tokens: u64,
}

/// 按 HTTP 状态码分类错误（共用实现见 [`crate::classify_status`]）。
fn classify_status(status: u16, text: &str) -> LlmError {
    crate::classify_status(status, text)
}

impl OpenAiAdapter {
    /// 由配置构造（默认请求头：Bearer + JSON）。
    pub fn new(config: OpenAiConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    /// 当前模型 id。
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// 组装请求体（OpenAI messages 投影 + stream 开关）。
    fn body(&self, request: &LlmRequest, streaming: bool) -> serde_json::Value {
        let messages: Vec<serde_json::Value> = request
            .messages
            .iter()
            .map(|message| match message {
                Message::System { content } => {
                    serde_json::json!({ "role": "system", "content": content })
                }
                Message::User(user) => {
                    // 带图片 → OpenAI 多部分 content（text + image_url）；纯文本 → 字符串
                    if user.has_images() {
                        let mut parts: Vec<serde_json::Value> = vec![serde_json::json!({
                            "type": "text",
                            "text": user.content
                        })];
                        for image in &user.images {
                            parts.push(serde_json::json!({
                                "type": "image_url",
                                "image_url": { "url": image }
                            }));
                        }
                        serde_json::json!({ "role": "user", "content": parts })
                    } else {
                        serde_json::json!({ "role": "user", "content": user.content })
                    }
                }
                Message::Assistant(assistant) => {
                    // 工具调用轮：assistant 历史必须带 tool_calls（OpenAI 协议，
                    // 否则后续 tool 结果消息会被网关拒绝/丢失）；无文本时 content 置 null
                    let tool_calls: Vec<serde_json::Value> = assistant
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::ToolUse { call } => Some(serde_json::json!({
                                "id": call.call_id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": call.arguments,
                                },
                            })),
                            ContentBlock::Text { .. } | ContentBlock::Thinking { .. } => None,
                        })
                        .collect();
                    let text = assistant.text();
                    let mut value = serde_json::json!({
                        "role": "assistant",
                        "content": if text.is_empty() {
                            serde_json::Value::Null
                        } else {
                            serde_json::json!(text)
                        },
                    });
                    if !tool_calls.is_empty() {
                        value["tool_calls"] = serde_json::json!(tool_calls);
                    }
                    value
                }
                Message::Tool(tool) => {
                    // OpenAI 协议：tool 消息必须带 tool_call_id（配对 assistant 的调用）
                    let mut value = serde_json::json!({ "role": "tool", "content": tool.content });
                    if let Some(call_id) = &tool.call_id {
                        value["tool_call_id"] = serde_json::json!(call_id);
                    }
                    value
                }
                Message::Custom { name, data } => serde_json::json!({
                    "role": "user",
                    "content": format!("[{name}]\n{data}")
                }),
            })
            .collect();
        let mut body = serde_json::json!({
            "model": self.config.model,
            "messages": messages,
            "stream": streaming,
        });
        if let Some(max_tokens) = self.config.max_tokens {
            body["max_tokens"] = serde_json::json!(max_tokens);
        }
        if !request.tools.is_empty() {
            body["tools"] = serde_json::json!(request.tools);
        }
        body
    }

    /// 构造一次 POST 请求（URL 非法在此失败）。
    fn build_request(
        &self,
        request: &LlmRequest,
        streaming: bool,
    ) -> Result<reqwest::Request, LlmError> {
        let url = format!(
            "{}/chat/completions",
            self.config.base_url.trim_end_matches('/')
        );
        let mut builder = self.client.post(&url).bearer_auth(&self.config.api_key);
        for (name, value) in &self.config.extra_headers {
            builder = builder.header(name, value);
        }
        builder
            .json(&self.body(request, streaming))
            .build()
            .map_err(|error| {
                LlmError::new(
                    LlmErrorCode::InvalidRequest,
                    format!("请求构造失败: {error}"),
                )
            })
    }
}

impl LlmAdapter for OpenAiAdapter {
    fn id(&self) -> &str {
        "openai"
    }

    fn input_content(&self) -> &[InputContent] {
        &self.config.input_content
    }

    fn stream(&self, request: &LlmRequest) -> LlmStream {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let client = self.client.clone();
        if !self.config.streaming {
            // 非流式模式：直接单次请求
            let single_req = match self.build_request(request, false) {
                Ok(req) => req,
                Err(error) => {
                    return Box::pin(futures::stream::once(async move { Err(error) }));
                }
            };
            tokio::spawn(async move {
                if let Err(error) = single_shot(&client, single_req, &tx).await {
                    let _ = tx.send(Err(error));
                }
            });
            return Box::pin(futures::stream::unfold(rx, |mut rx| async move {
                rx.recv().await.map(|item| (item, rx))
            }));
        }
        let streaming_req = match self.build_request(request, true) {
            Ok(req) => req,
            Err(error) => {
                return Box::pin(futures::stream::once(async move { Err(error) }));
            }
        };
        let single_req = match self.build_request(request, false) {
            Ok(req) => req,
            Err(error) => {
                return Box::pin(futures::stream::once(async move { Err(error) }));
            }
        };
        tokio::spawn(async move {
            // 错误不进 stderr：一律作为流内 Err 交付给消费方
            if let Err(error) = run_request(client, streaming_req, single_req, tx.clone()).await {
                let _ = tx.send(Err(error));
            }
        });
        Box::pin(futures::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        }))
    }
}

/// 后台任务：流式优先；**可重试失败**（服务端/限流/网络/未分类）且尚无产出 → 非流式兜底。
async fn run_request(
    client: reqwest::Client,
    streaming_req: reqwest::Request,
    single_req: reqwest::Request,
    tx: tokio::sync::mpsc::UnboundedSender<Result<StreamChunk, LlmError>>,
) -> Result<(), LlmError> {
    let mut sent = 0usize;
    match stream_once(&client, streaming_req, &tx, &mut sent).await {
        Ok(()) => Ok(()),
        Err(error) => {
            if sent == 0 && error.is_retryable() {
                match single_shot(&client, single_req, &tx).await {
                    Ok(()) => Ok(()),
                    Err(fallback) => Err(LlmError {
                        code: fallback.code,
                        message: format!("{}（非流式兜底同样失败）", fallback.message),
                        facts: fallback.facts,
                    }),
                }
            } else {
                Err(error)
            }
        }
    }
}

/// 流式执行：SSE 逐行解析并转发；失败带稳定错误码（[`LlmErrorCode`]）。
async fn stream_once(
    client: &reqwest::Client,
    req: reqwest::Request,
    tx: &tokio::sync::mpsc::UnboundedSender<Result<StreamChunk, LlmError>>,
    sent: &mut usize,
) -> Result<(), LlmError> {
    let response = client
        .execute(req)
        .await
        .map_err(|error| LlmError::new(LlmErrorCode::Network, format!("请求失败: {error}")))?;
    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(classify_status(status.as_u16(), &text));
    }

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut finished = false;
    let mut eof = false;
    // 工具调用分片：按 index 归组（id/name 取首个片段，arguments 拼接）
    let mut tool_acc: Vec<Option<(String, String, String)>> = Vec::new();
    // 最后见到的 usage（流尾工具块携带）
    let mut usage_tail: Option<TokenUsage> = None;
    while !finished {
        if let Some(pos) = buffer.find('\n') {
            // 逐行消费 SSE；error 体可能不带 `data:` 前缀（裸 JSON）
            let line: String = buffer[..pos].trim_end_matches('\r').to_string();
            buffer.drain(..=pos);
            let data = match line.strip_prefix("data:") {
                Some(data) => data.trim(),
                None => line.trim(),
            };
            if data.is_empty() {
                continue;
            }
            if data == "[DONE]" {
                finished = true;
                continue;
            }
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(data)
                && value.get("type").and_then(|t| t.as_str()) == Some("error")
            {
                return Err(LlmError::new(
                    LlmErrorCode::Server,
                    format!("服务端错误: {}", truncate(data, 300)),
                ));
            }
            let chunk: SseChunk = match serde_json::from_str(data) {
                Ok(chunk) => chunk,
                Err(_) => continue, // keepalive/注释行忽略
            };
            let mut text = String::new();
            let mut reasoning = String::new();
            for choice in &chunk.choices {
                if let Some(content) = &choice.delta.content {
                    text.push_str(content);
                }
                if let Some(thought) = &choice.delta.reasoning_content {
                    reasoning.push_str(thought);
                }
                for call in &choice.delta.tool_calls {
                    let index = call.index.unwrap_or(0);
                    while tool_acc.len() <= index {
                        tool_acc.push(None);
                    }
                    let slot = tool_acc[index]
                        .get_or_insert_with(|| (String::new(), String::new(), String::new()));
                    if let Some(id) = &call.id
                        && slot.0.is_empty()
                    {
                        slot.0.push_str(id);
                    }
                    if let Some(name) = &call.function.name
                        && slot.1.is_empty()
                    {
                        slot.1.push_str(name);
                    }
                    if let Some(arguments) = &call.function.arguments {
                        slot.2.push_str(arguments);
                    }
                }
            }
            let usage = chunk.usage.map(|usage| TokenUsage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
            });
            if usage.is_some() {
                usage_tail = usage;
            }
            *sent += 1;
            // 推理与正文分开流式（thinking 一到达即转发，不缓冲不丢弃；
            // 消费方自行决定展示；usage 独立成块，消费方跳过）
            if !reasoning.is_empty()
                && tx
                    .send(Ok(StreamChunk {
                        delta: ChunkDelta::Thinking { text: reasoning },
                        usage: None,
                    }))
                    .is_err()
            {
                return Ok(()); // 接收方已丢弃（提前取消）
            }
            if !text.is_empty() {
                if tx
                    .send(Ok(StreamChunk {
                        delta: ChunkDelta::Text { text },
                        usage,
                    }))
                    .is_err()
                {
                    return Ok(()); // 接收方已丢弃（提前取消）
                }
            } else if usage.is_some()
                && tx
                    .send(Ok(StreamChunk {
                        delta: ChunkDelta::Text {
                            text: String::new(),
                        },
                        usage,
                    }))
                    .is_err()
            {
                return Ok(()); // 接收方已丢弃（提前取消）
            }
        } else if eof {
            // EOF：残留无换行尾行（裸 JSON error 体等）补一个换行再处理一次
            if !buffer.is_empty() {
                buffer.push('\n');
                continue;
            }
            break;
        } else {
            match stream.next().await {
                Some(Ok(bytes)) => {
                    buffer.push_str(&String::from_utf8_lossy(&bytes));
                }
                Some(Err(error)) => {
                    return Err(LlmError::new(
                        LlmErrorCode::Network,
                        format!("流读取失败: {error}"),
                    ));
                }
                None => eof = true,
            }
        }
    }
    // 流尾：工具调用分片 → 合成 ToolUse 块
    let tools: Vec<ToolCall> = tool_acc
        .into_iter()
        .flatten()
        .filter(|(_, name, arguments)| !name.is_empty() || !arguments.is_empty())
        .map(|(call_id, name, arguments)| ToolCall {
            call_id,
            name,
            arguments,
        })
        .collect();
    for (i, call) in tools.iter().enumerate() {
        *sent += 1;
        let _ = tx.send(Ok(StreamChunk {
            delta: ChunkDelta::ToolUse { call: call.clone() },
            usage: if i + 1 == tools.len() {
                usage_tail
            } else {
                None
            },
        }));
    }
    // 终结分片：显式结束原因（有工具块 → ToolCalls，否则 Stop；对齐 dsh finish chunk）
    let _ = tx.send(Ok(StreamChunk {
        delta: ChunkDelta::Finish {
            reason: if tools.is_empty() {
                FinishReason::Stop
            } else {
                FinishReason::ToolCalls
            },
        },
        usage: None,
    }));
    Ok(())
}

/// 非流式兜底：单次请求，整段内容合成一个 chunk。
async fn single_shot(
    client: &reqwest::Client,
    req: reqwest::Request,
    tx: &tokio::sync::mpsc::UnboundedSender<Result<StreamChunk, LlmError>>,
) -> Result<(), LlmError> {
    let response = client
        .execute(req)
        .await
        .map_err(|error| LlmError::new(LlmErrorCode::Network, format!("请求失败: {error}")))?;
    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(classify_status(status.as_u16(), &text));
    }
    let raw = response
        .text()
        .await
        .map_err(|error| LlmError::new(LlmErrorCode::Network, format!("响应读取失败: {error}")))?;
    let completion: Completion = serde_json::from_str(&raw).map_err(|error| {
        LlmError::new(
            LlmErrorCode::Protocol,
            format!("非流式响应不是合法 JSON: {error}"),
        )
    })?;
    // 逐 choice 收集：思考块（reasoning_content）+ 文本块 + 工具调用块
    let mut thinking = String::new();
    let mut text = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut usage = None;
    for choice in &completion.choices {
        let reasoning = choice
            .message
            .reasoning_content
            .as_deref()
            .filter(|c| !c.trim().is_empty());
        if let Some(reasoning) = reasoning {
            thinking.push_str(reasoning);
        }
        // content 优先（空串视同缺失）；缺省时不再把 reasoning 混入文本（走 Thinking 块）
        let content = choice
            .message
            .content
            .as_deref()
            .filter(|c| !c.trim().is_empty());
        if let Some(content) = content {
            text.push_str(content);
        }
        for call in &choice.message.tool_calls {
            let name = call.function.name.clone().unwrap_or_default();
            let arguments = call.function.arguments.clone().unwrap_or_default();
            if name.is_empty() && arguments.is_empty() {
                continue;
            }
            tool_calls.push(ToolCall {
                call_id: call.id.clone().unwrap_or_default(),
                name,
                arguments,
            });
        }
    }
    if let Some(u) = completion.usage {
        usage = Some(TokenUsage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
        });
    }
    // 镜像流式顺序：思考 → 文本 → 工具；usage 挂在最后一块
    let mut chunks: Vec<Result<StreamChunk, LlmError>> = Vec::new();
    if !thinking.is_empty() {
        chunks.push(Ok(StreamChunk {
            delta: ChunkDelta::Thinking { text: thinking },
            usage: None,
        }));
    }
    if !text.is_empty() {
        chunks.push(Ok(StreamChunk {
            delta: ChunkDelta::Text { text },
            usage: None,
        }));
    }
    for call in tool_calls.iter() {
        chunks.push(Ok(StreamChunk {
            delta: ChunkDelta::ToolUse { call: call.clone() },
            usage: None,
        }));
    }
    if let Some(Ok(last)) = chunks.last_mut() {
        last.usage = usage;
    }
    // 终结分片：显式结束原因（有工具调用 → ToolCalls，否则 Stop）
    chunks.push(Ok(StreamChunk {
        delta: ChunkDelta::Finish {
            reason: if tool_calls.is_empty() {
                FinishReason::Stop
            } else {
                FinishReason::ToolCalls
            },
        },
        usage: None,
    }));
    for chunk in chunks {
        if tx.send(chunk).is_err() {
            return Ok(()); // 接收方已丢弃（提前取消）
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LlmRequest, UserMessage};

    /// extra_headers 必须逐条落到请求上（opencode go 的会话头依赖这条）。
    #[test]
    fn extra_headers_land_on_request() {
        let adapter = OpenAiAdapter::new(OpenAiConfig {
            base_url: "https://gateway.test/v1".into(),
            api_key: "k".into(),
            model: "m".into(),
            streaming: false,
            max_tokens: Some(16),
            input_content: vec![InputContent::Text],
            extra_headers: vec![
                ("x-opencode-session".to_string(), "sess-1".to_string()),
                ("User-Agent".to_string(), "test-agent/1.0".to_string()),
            ],
        });
        let request = LlmRequest {
            system: None,
            messages: vec![Message::User(UserMessage::new("hi"))],
            tools: Vec::new(),
        };
        let built = adapter.build_request(&request, false).unwrap();
        let headers = built.headers();
        assert_eq!(headers.get("x-opencode-session").unwrap(), "sess-1");
        assert_eq!(headers.get("user-agent").unwrap(), "test-agent/1.0");
        assert!(headers.get("authorization").is_some(), "鉴权头仍在");
    }
}
