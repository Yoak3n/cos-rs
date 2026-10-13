//! P6 验收：A 形态 DoD —— demo 端到端快照（逐事件一致，keyless snapshot 等价物）、
//! 不变量全过、JSONL 可重放、卸载顺序可审计、--dump-config 与装载一致。
//!
//! LLM 链路：cos 不再内置 Provider——测试经 `--llm-*`（RunConfig.llm）把真实
//! opencode 适配器指向本地回环 chat/completions 服务器（cos-test-support），
//! 以真实适配器协议离线确定性驱动（脚本：tool_use → 文本回复）。

use cos::{
    run,
    schema::{LlmConfig, RunConfig},
};
use cos_test_support::{ChatReply, ScriptedChatServer};

fn demo_config(base_url: String) -> RunConfig {
    RunConfig {
        config_path: Some(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/demo.yml").to_string()),
        dump_config: false,
        session_id: "demo-e2e".into(),
        prompt: Some("帮我记一条演示 todo".into()),
        session_path: None,
        cancel: None,
        llm: Some(LlmConfig {
            base_url,
            api_key: "test-key".into(),
            model: "test-model".into(),
            streaming: false,
        }),
        agent_llm: None,
        agent_driver: None,
        patch_files: Vec::new(),
    }
}

#[tokio::test]
async fn demo_end_to_end_snapshot_invariants_replay_unload() {
    let session_path = std::env::temp_dir().join(format!("cos-demo-{}.jsonl", std::process::id()));
    let server = ScriptedChatServer::spawn(vec![
        ChatReply::ToolUse {
            id: "demo-call-1".into(),
            name: "todo_write".into(),
            arguments: r#"{"todos":[{"content":"演示任务：验证 A 形态","status":"in_progress"}]}"#
                .into(),
        },
        ChatReply::Text("已记录演示任务。".into()),
    ])
    .await;
    let mut config = demo_config(format!("http://127.0.0.1:{}/v1", server.port));
    config.session_path = Some(session_path.to_string_lossy().into_owned());

    let report = run(config).await.unwrap();

    // 1. 快照：逐事件类型一致（时间戳除外 —— keyless snapshot 等价物）
    let types: Vec<&str> = report
        .events
        .iter()
        .map(|event| event.data.type_name())
        .collect();
    // 非流式适配器：tool_use 与文本各合成 1 个 chunk（流式 mock 曾按字符切 8 块）
    let expected: Vec<&str> = vec![
        "turn/start",
        "step/start",
        "user/message",
        "request/header",
        "assistant/chunk", // tool_use
        "assistant/message",
        "tool/call",
        "todo/write", // 工具执行期间的会话态（因果链内）
        "tool/result",
        "step/end",
        "step/start", // 结果回流 step
        "request/header",
        "assistant/chunk", // 文本（单 chunk）
        "assistant/message",
        "step/end",
        "turn/end",
    ];
    assert_eq!(types, expected);

    // 2. derive_messages 重放一致（含完整 call/result 对 + 回流回复）
    assert_eq!(report.messages.len(), 4);
    match &report.messages[0] {
        cos_llm::Message::User(user) => assert_eq!(user.content, "帮我记一条演示 todo"),
        other => panic!("期望 User，实际 {other:?}"),
    }
    match &report.messages[2] {
        cos_llm::Message::Tool(tool) => assert_eq!(tool.content, "已写入 1 条任务"),
        other => panic!("期望 Tool，实际 {other:?}"),
    }
    // 任务内容在 todo/write 会话事件里（工具经因果链写入）
    let todo_written = report.events.iter().any(|event| {
        matches!(&event.data, cos_session::SessionEventData::TodoWrite { todos }
            if todos.iter().any(|item| item.content.contains("演示任务")))
    });
    assert!(todo_written, "会话日志应含 todo/write 且内容为演示任务");
    match &report.messages[3] {
        cos_llm::Message::Assistant(assistant) => {
            assert_eq!(assistant.text(), "已记录演示任务。")
        }
        other => panic!("期望 Assistant，实际 {other:?}"),
    }

    // 3. 不变量全过（模型可见 ⟺ 已记录、seq 单调、边界配对、call/result 配对）
    assert!(report.violations.is_empty(), "{:?}", report.violations);

    // 4. 卸载顺序：apply 逆序（审计）
    assert_eq!(
        report.unload_order,
        vec!["rpc", "bash", "todo", "opencode-provider"]
    );
    assert!(report.services_after_unload, "todo 服务应已随卸载反注册");

    // 5. JSONL 落盘（重放一致性已由 run 内部校验；落盘走压缩口径：
    //    2 条被 message 盖住的 chunk 不落 → header + 14 事件）
    let text = std::fs::read_to_string(&session_path).unwrap();
    assert_eq!(text.lines().count(), 15);
    assert!(
        !text.contains("assistant/chunk"),
        "被 message 盖住的 chunk 不应落盘"
    );
    // header 的创建时间是真的（回归点：收尾落盘曾把它写死 0）
    let (header, _) = cos_session::load_jsonl(&session_path).unwrap();
    assert!(header.created_at_ms > 0, "收尾落盘不该把创建时间冲成 0");
    assert_eq!(header.id, "demo-e2e");

    server.join().await;
}

#[tokio::test]
async fn dump_config_matches_loaded_order() {
    // dump 路径不装配（不发 LLM 请求）——无需服务器，llm 配置被忽略
    let mut config = demo_config("http://127.0.0.1:1/v1".into());
    config.dump_config = true;
    let report = run(config).await.unwrap();

    let dump: serde_json::Value =
        serde_json::from_str(report.dump.as_deref().expect("dump 输出")).unwrap();
    let entries = dump.as_array().expect("计划应为数组");
    assert_eq!(entries.len(), 4);
    let names: Vec<&str> = entries
        .iter()
        .map(|entry| entry["name"].as_str().unwrap())
        .collect();
    // 与装载顺序一致（demo.yml 无依赖 → 清单序即拓扑序）
    assert_eq!(names, vec!["opencode-provider", "todo", "bash", "rpc"]);
    assert_eq!(entries[0]["id"], "opencode-provider"); // id 缺省 = name
    assert_eq!(entries[0]["config"], serde_json::Value::Null);
}

#[tokio::test]
async fn unknown_plugin_config_fails_loud() {
    let root_path = std::env::temp_dir().join(format!("cos-bad-{}.yml", std::process::id()));
    std::fs::write(&root_path, "- name: nope\n").unwrap();
    let server = ScriptedChatServer::spawn(vec![]).await;
    let mut config = demo_config(format!("http://127.0.0.1:{}/v1", server.port));
    config.config_path = Some(root_path.to_string_lossy().into_owned());
    let result = run(config).await;
    assert!(matches!(result, Err(ref error) if error.to_string().contains("nope")));
    server.join().await;
}

/// 关键组件缺失：无任何 LLM 配置 → 启动失败，错误信息给出接入方式。
#[tokio::test]
async fn no_llm_config_fails_startup_with_guidance() {
    let root_path = std::env::temp_dir().join(format!("cos-nollm-{}.yml", std::process::id()));
    std::fs::write(&root_path, "- name: todo\n").unwrap();
    let mut config = demo_config("http://127.0.0.1:1/v1".into()); // 端口无所谓——装配即失败
    config.config_path = Some(root_path.to_string_lossy().into_owned());
    config.llm = None;
    let error = match run(config).await {
        Err(error) => error.to_string(),
        Ok(_) => panic!("无 LLM 配置应启动失败"),
    };
    assert!(error.contains("未配置 LLM"), "实际: {error}");
    assert!(
        error.contains("--llm-base-url"),
        "应提示命令行接入: {error}"
    );
    assert!(error.contains("plugin-llm"), "应提示 yml 接入: {error}");
    let _ = std::fs::remove_file(&root_path);
}
