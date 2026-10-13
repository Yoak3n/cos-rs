//! 本地回环 HTTP `chat/completions` 服务器（非流式，OpenAI 响应形状）。
//!
//! 用途：e2e 经 `--llm-*` 把真实 cos 二进制指向本服务器——真实适配器协议
//! （cos-llm 的 openai feature 非流式解析）+ 真实 CLI 链路，离线确定性测试。

use std::sync::Arc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

/// 一条脚本化回复（按请求序号依次下发）。
#[derive(Debug, Clone)]
pub enum ChatReply {
    /// 纯文本回复。
    Text(String),
    /// 工具调用回复（content=null + tool_calls；`arguments` 为 JSON 文本）。
    ToolUse {
        /// 调用 id。
        id: String,
        /// 工具名。
        name: String,
        /// 原始 JSON 参数字符串。
        arguments: String,
    },
}

/// 已启动的脚本化服务器（drop 后任务继续跑完剩余连接；await [`ScriptedChatServer::join`] 收束）。
pub struct ScriptedChatServer {
    /// 监听端口（`--llm-base-url http://127.0.0.1:{port}/v1`）。
    pub port: u16,
    handle: Arc<JoinHandle<()>>,
}

impl ScriptedChatServer {
    /// 启动服务器：依次为每个回复接受一个连接并应答（非流式 JSON）。
    pub async fn spawn(replies: Vec<ChatReply>) -> Self {
        Self::spawn_with_key(replies, None).await
    }

    /// 启动服务器并校验每个请求的 `Authorization: Bearer <key>`（验证插件配置的 api_key
    /// 真的被适配器用上；None = 不校验）。
    pub async fn spawn_with_key(replies: Vec<ChatReply>, expect_api_key: Option<String>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = tokio::spawn(async move {
            for reply in replies {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut buf = vec![0u8; 65536];
                let n = socket.read(&mut buf).await.unwrap();
                let request = String::from_utf8_lossy(&buf[..n]).to_string();
                assert!(
                    request.contains("chat/completions"),
                    "请求路径不符: {request}"
                );
                if let Some(key) = &expect_api_key {
                    assert!(
                        request
                            .to_lowercase()
                            .contains(&format!("authorization: bearer {key}")),
                        "请求应携带 Authorization: Bearer {key}: {request}"
                    );
                }
                let payload = match &reply {
                    ChatReply::Text(text) => serde_json::json!({
                        "choices": [{
                            "index": 0,
                            "message": { "role": "assistant", "content": text },
                        }],
                        "usage": { "prompt_tokens": 10, "completion_tokens": 4 },
                    }),
                    ChatReply::ToolUse {
                        id,
                        name,
                        arguments,
                    } => serde_json::json!({
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": null,
                                "tool_calls": [{
                                    "id": id,
                                    "type": "function",
                                    "function": { "name": name, "arguments": arguments },
                                }],
                            },
                        }],
                        "usage": { "prompt_tokens": 10, "completion_tokens": 4 },
                    }),
                };
                let body = serde_json::to_string(&payload).unwrap_or_default();
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{body}"
                );
                socket.write_all(response.as_bytes()).await.unwrap();
                socket.shutdown().await.unwrap();
            }
        });
        ScriptedChatServer {
            port,
            handle: Arc::new(handle),
        }
    }

    /// 等待服务器收束（所有脚本回复已下发；断言失败会 panic 到测试线程）。
    pub async fn join(self) {
        let handle = Arc::into_inner(self.handle).expect("唯一引用");
        handle.await.unwrap();
    }
}

/// 同步启动（非 async 上下文，如同步 `#[test]`）：独立 OS 线程 + 独立 current-thread runtime。
///
/// 返回 `(port, thread_handle)`；服务器任务跑完脚本回复后线程自然结束。
pub fn spawn_sync(replies: Vec<ChatReply>) -> (u16, std::thread::JoinHandle<()>) {
    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        rt.block_on(async {
            let server = ScriptedChatServer::spawn(replies).await;
            tx.send(server.port).expect("port 信道");
            server.join().await;
        });
    });
    let port = rx.recv().expect("服务器端口");
    (port, handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cos_llm::{InputContent, LlmAdapter, LlmRequest, Message, UserMessage};
    use futures::StreamExt;

    /// 用真实适配器（cos-llm openai feature）打桩：非流式文本回复全链路。
    #[tokio::test]
    async fn serves_text_reply_to_opencode_adapter() {
        let server = ScriptedChatServer::spawn(vec![ChatReply::Text("你好，世界".into())]).await;
        let adapter = adapter_for(server.port);
        let mut stream = adapter.stream(&request());
        let mut text = String::new();
        while let Some(item) = stream.next().await {
            let chunk = item.unwrap();
            if let cos_llm::ChunkDelta::Text { text: delta } = chunk.delta {
                text.push_str(&delta);
            }
        }
        assert_eq!(text, "你好，世界");
        server.join().await;
    }

    /// 工具调用回复 → 适配器产出 ToolUse 块。
    #[tokio::test]
    async fn serves_tool_use_reply_to_opencode_adapter() {
        let server = ScriptedChatServer::spawn(vec![ChatReply::ToolUse {
            id: "call_1".into(),
            name: "todo_write".into(),
            arguments: r#"{"todos":[]}"#.into(),
        }])
        .await;
        let adapter = adapter_for(server.port);
        let mut stream = adapter.stream(&request());
        let chunk = stream.next().await.unwrap().unwrap();
        match chunk.delta {
            cos_llm::ChunkDelta::ToolUse { call } => {
                assert_eq!(call.name, "todo_write");
                assert_eq!(call.arguments, r#"{"todos":[]}"#);
            }
            other => panic!("期望 ToolUse，实际 {other:?}"),
        }
        server.join().await;
    }

    fn adapter_for(port: u16) -> cos_llm::OpenAiAdapter {
        cos_llm::OpenAiAdapter::new(cos_llm::OpenAiConfig {
            base_url: format!("http://127.0.0.1:{port}/v1"),
            api_key: "test-key".into(),
            model: "test-model".into(),
            streaming: false,
            max_tokens: Some(2048),
            input_content: vec![InputContent::Text],
            extra_headers: Vec::new(),
        })
    }

    fn request() -> LlmRequest {
        LlmRequest {
            system: None,
            messages: vec![Message::User(UserMessage::new("你好"))],
            tools: vec![],
        }
    }
}
