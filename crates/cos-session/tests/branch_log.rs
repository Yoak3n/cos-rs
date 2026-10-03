//! 树状会话验收：分支隔离、折叠回流、树投影、错误边界、JSONL 往返与旧日志兼容。

use cos_llm::{AssistantMessage, ContentBlock, Message, UserMessage};
use cos_session::{
    SESSION_FORMAT_VERSION, Session, SessionError, SessionEventData, SessionHeader, load_jsonl,
    save_jsonl,
};

/// 在主干上落一轮「用户问 + 助手答」。
fn trunk_turn(session: &Session, question: &str, answer: &str, base: u64) {
    session.append_at(
        SessionEventData::UserMessage(UserMessage::new(question)),
        base,
    );
    session.append_at(
        SessionEventData::AssistantMessage {
            turn: 1,
            step: 1,
            message: AssistantMessage::new(vec![ContentBlock::Text {
                text: answer.into(),
            }]),
            usage: None,
        },
        base + 1,
    );
}

/// 把模型可见历史压成「用户说了什么」的序列，断言用。
fn user_texts(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|m| match m {
            Message::User(user) => Some(user.content.clone()),
            Message::Assistant(a) => Some(format!("[答]{}", a.text())),
            _ => None,
        })
        .collect()
}

/// 没有任何 `branch/open` 的会话，投影与从前完全一致。
#[test]
fn unbranched_session_projects_as_before() {
    let session = Session::new("s1");
    trunk_turn(&session, "问一", "答一", 100);
    assert!(session.current_branch().is_none());
    assert_eq!(user_texts(&session.derive_messages()), ["问一", "[答]答一"]);
    let tree = session.branch_tree();
    assert!(tree.is_trunk());
    assert!(tree.children.is_empty());
    assert_eq!(tree.count(), 1);
}

/// 支线看得到自己从哪儿岔出来的上下文（祖先链到分叉点）。
#[test]
fn branch_sees_ancestors_up_to_the_fork() {
    let session = Session::new("s1");
    trunk_turn(&session, "问一", "答一", 100); // seq 1,2
    let fork = session.last_seq(); // 2

    let branch = session.open_branch("装饰器", fork).unwrap();
    assert_eq!(branch, "br_1");
    assert_eq!(session.current_branch().as_deref(), Some("br_1"));
    trunk_turn(&session, "追问闭包", "闭包是…", 200); // seq 4,5（归属 br_1）

    assert_eq!(
        user_texts(&session.derive_messages()),
        ["问一", "[答]答一", "追问闭包", "[答]闭包是…"]
    );
}

/// **岔路物理隔离**：主干看不到支线内容；支线也看不到父分支在分岔之后的新内容。
#[test]
fn branch_and_trunk_are_physically_isolated() {
    let session = Session::new("s1");
    trunk_turn(&session, "问一", "答一", 100); // seq 1,2
    let fork = session.last_seq();

    session.open_branch("岔路", fork).unwrap();
    trunk_turn(&session, "支线问", "支线答", 200); // seq 4,5

    // 回主干续聊
    session.enter_branch(None).unwrap();
    trunk_turn(&session, "主干又问", "主干又答", 300); // seq 7,8

    // 主干视野：只有主干内容，支线一个字都不回流
    assert_eq!(
        user_texts(&session.derive_messages()),
        ["问一", "[答]答一", "主干又问", "[答]主干又答"]
    );

    // 支线视野：分叉点之前的祖先 + 支线自己；主干在分岔之后的新内容看不见
    let branch_view = session.derive_branch_messages(Some("br_1"));
    assert_eq!(
        user_texts(&branch_view),
        ["问一", "[答]答一", "支线问", "[答]支线答"]
    );
}

/// 折叠支线：写摘要、游标交还父分支、已折叠不能再折。
#[test]
fn close_branch_folds_back_to_parent() {
    let session = Session::new("s1");
    trunk_turn(&session, "问一", "答一", 100);
    let fork = session.last_seq();
    session.open_branch("装饰器", fork).unwrap();
    trunk_turn(&session, "支线问", "支线答", 200);

    let closed = session
        .close_branch("装饰器 = 定义期把函数换成返回值")
        .unwrap();
    assert_eq!(closed, "br_1");
    assert!(session.current_branch().is_none(), "游标应回到主干");

    // 摘要进树投影
    let tree = session.branch_tree();
    assert_eq!(tree.children.len(), 1);
    assert_eq!(
        tree.children[0].closed.as_deref(),
        Some("装饰器 = 定义期把函数换成返回值")
    );
    assert_eq!(tree.children[0].label, "装饰器");
    assert_eq!(tree.children[0].fork_seq, fork);

    // 主干续聊仍不受影响
    trunk_turn(&session, "主干又问", "主干又答", 300);
    assert_eq!(
        user_texts(&session.derive_messages()),
        ["问一", "[答]答一", "主干又问", "[答]主干又答"]
    );

    // 已折叠的分支不能再折
    session.enter_branch(Some("br_1")).unwrap();
    let err = session.close_branch("再折一次").unwrap_err();
    assert!(matches!(err, SessionError::Invalid(_)), "实际 {err:?}");
}

/// 嵌套支线：支线里再开支线，各自的祖先链正确。
#[test]
fn nested_branches_chain_their_ancestors() {
    let session = Session::new("s1");
    trunk_turn(&session, "问一", "答一", 100); // seq 1,2

    session.open_branch("一层", 2).unwrap();
    trunk_turn(&session, "一层的问", "一层的答", 200); // seq 4,5
    let inner_fork = session.last_seq(); // 5

    session.open_branch("二层", inner_fork).unwrap();
    trunk_turn(&session, "二层的问", "二层的答", 300); // seq 7,8

    assert_eq!(
        user_texts(&session.derive_messages()),
        [
            "问一",
            "[答]答一",
            "一层的问",
            "[答]一层的答",
            "二层的问",
            "[答]二层的答"
        ]
    );

    // 一层的视野不该有二层的内容
    assert_eq!(
        user_texts(&session.derive_branch_messages(Some("br_1"))),
        ["问一", "[答]答一", "一层的问", "[答]一层的答"]
    );

    // 树投影：主干 → 一层 → 二层
    let tree = session.branch_tree();
    assert_eq!(tree.count(), 3);
    assert_eq!(tree.children[0].label, "一层");
    assert_eq!(tree.children[0].children[0].label, "二层");
    assert_eq!(tree.children[0].children[0].parent.as_deref(), Some("br_1"));
}

/// 边界：分叉点越界、主干折叠、进不存在的分支。
#[test]
fn invalid_branch_operations_are_rejected() {
    let session = Session::new("s1");
    trunk_turn(&session, "问一", "答一", 100); // last_seq = 2

    let err = session.open_branch("越界", 99).unwrap_err();
    assert!(matches!(err, SessionError::Invalid(_)), "实际 {err:?}");

    let err = session.close_branch("主干不能折").unwrap_err();
    assert!(matches!(err, SessionError::Invalid(_)), "实际 {err:?}");

    let err = session.enter_branch(Some("br_404")).unwrap_err();
    assert!(matches!(err, SessionError::Invalid(_)), "实际 {err:?}");

    // 分叉点可以正好落在最后一条事件上（含）
    assert!(session.open_branch("正好", 2).is_ok());
}

/// JSONL 往返：分支标记与开关事件逐字节存回。
#[test]
fn branch_log_roundtrips_through_jsonl() {
    let session = Session::new("s1");
    trunk_turn(&session, "问一", "答一", 100);
    session.open_branch("装饰器", 2).unwrap();
    trunk_turn(&session, "支线问", "支线答", 200);
    session.close_branch("摘要").unwrap();

    let header = SessionHeader {
        version: SESSION_FORMAT_VERSION,
        id: session.id().to_string(),
        created_at_ms: 7,
        cwd: None,
    };
    let path = std::env::temp_dir().join(format!("cos-branch-{}.jsonl", std::process::id()));
    save_jsonl(&session, &header, &path).unwrap();
    let original = std::fs::read(&path).unwrap();

    let (_, events) = load_jsonl(&path).unwrap();
    assert_eq!(events, session.events().to_vec());
    let restored = Session::from_events("s1", events);
    assert_eq!(restored.branch_tree(), session.branch_tree());
    assert_eq!(
        user_texts(&restored.derive_branch_messages(Some("br_1"))),
        ["问一", "[答]答一", "支线问", "[答]支线答"]
    );

    save_jsonl(&restored, &header, &path).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), original, "应逐字节一致");

    // 恢复后接着开分支：id 不撞车
    let resumed = Session::from_events("s1", session.events());
    assert_eq!(resumed.open_branch("新支线", 0).unwrap(), "br_2");
}

/// 旧日志（没有 `branch` 字段）照读，读作主干。
#[test]
fn legacy_log_without_branch_field_still_loads() {
    let path = std::env::temp_dir().join(format!("cos-legacy-{}.jsonl", std::process::id()));
    let header =
        format!("{{\"version\":{SESSION_FORMAT_VERSION},\"id\":\"old\",\"createdAt\":1}}\n");
    let user = serde_json::json!({
        "seq": 1,
        "time": 10,
        "type": "user/message",
        "data": { "content": "老日志", "images": [] }
    });
    std::fs::write(&path, format!("{header}{user}\n")).unwrap();

    let (_, events) = load_jsonl(&path).unwrap();
    assert_eq!(events.len(), 1);
    assert!(events[0].branch.is_none(), "缺字段应读作主干");
    let session = Session::from_events("old", events);
    assert_eq!(user_texts(&session.derive_messages()), ["老日志"]);
}

/// 事件 wire 形状：分支标记在信封顶层，`type`/`data` 不受影响。
#[test]
fn branch_stamp_sits_on_the_envelope() {
    let session = Session::new("s1");
    session.append_at(SessionEventData::UserMessage(UserMessage::new("主干")), 1);
    session.open_branch("支线", 1).unwrap();
    session.append_at(SessionEventData::UserMessage(UserMessage::new("支线")), 2);

    let events = session.events();
    let trunk = serde_json::to_value(&events[0]).unwrap();
    assert_eq!(trunk["type"], "user/message");
    assert!(trunk.get("branch").is_none(), "主干不打标记（省字节）");

    let open = serde_json::to_value(&events[1]).unwrap();
    assert_eq!(open["type"], "branch/open");
    assert_eq!(open["branch"], "br_1");
    assert_eq!(open["data"]["branchId"], "br_1");
    assert_eq!(open["data"]["parentSeq"], 1);
    assert_eq!(open["data"]["label"], "支线");

    let inner = serde_json::to_value(&events[2]).unwrap();
    assert_eq!(inner["branch"], "br_1");
}

/// **折叠不是终态**：折叠后再进去说话，这一支算「续上了」，可以再次折叠。
#[test]
fn a_folded_branch_can_be_resumed_and_folded_again() {
    let session = Session::new("s1");
    trunk_turn(&session, "问一", "答一", 100);
    let fork = session.last_seq();
    session.open_branch("装饰器", fork).unwrap();
    trunk_turn(&session, "支线问", "支线答", 200);

    session.close_branch("第一版结论").unwrap();
    assert!(session.current_branch().is_none(), "游标回主干");

    // 折叠后进树投影：没续过 → 不是 resumed，摘要 = 第一版
    let tree = session.branch_tree();
    assert_eq!(tree.children[0].closed.as_deref(), Some("第一版结论"));
    assert!(!tree.children[0].resumed);

    // 再进去说一轮：事件照常写入该分支
    session.enter_branch(Some("br_1")).unwrap();
    trunk_turn(&session, "追问闭包", "闭包是…", 300);

    let tree = session.branch_tree();
    assert!(tree.children[0].resumed, "折叠之后有新内容 → 这一支是活的");
    assert_eq!(
        tree.children[0].closed.as_deref(),
        Some("第一版结论"),
        "摘要仍停在最后一次折叠那一刻"
    );

    // 活的分支可以再折一次：摘要取最新
    session.close_branch("第二版结论").unwrap();
    let tree = session.branch_tree();
    assert_eq!(tree.children[0].closed.as_deref(), Some("第二版结论"));
    assert!(!tree.children[0].resumed, "刚折完，又静止了");

    // 折完又没动就再折 → 拒（老守卫保住了）
    session.enter_branch(Some("br_1")).unwrap();
    let err = session.close_branch("第三次").unwrap_err();
    assert!(matches!(err, SessionError::Invalid(_)), "实际 {err:?}");
}

/// 折叠后在它下面再开支线，父分支也算「续上了」（冻结父不该挂着活子）。
#[test]
fn a_folded_branch_is_resumed_when_a_child_branch_appears() {
    let session = Session::new("s1");
    trunk_turn(&session, "问一", "答一", 100);
    let fork = session.last_seq();
    session.open_branch("一层", fork).unwrap();
    trunk_turn(&session, "一层的问", "一层的答", 200);
    session.close_branch("一层结论").unwrap();

    // 折叠一层之后，在它下面开二层。
    // 注意游标此刻在主干（折叠把它交还了父分支）——要挂「一层 → 二层」必须显式进去，
    // 这正是 Session::open_branch 要 parent 而不是靠游标的原因。
    session.enter_branch(Some("br_1")).unwrap();
    let inner_fork = session.last_seq();
    session.open_branch("二层", inner_fork).unwrap();
    trunk_turn(&session, "二层的问", "二层的答", 300);

    let tree = session.branch_tree();
    let outer = &tree.children[0];
    assert!(outer.resumed, "子分支有动静 → 父分支不是冻结态");
    assert_eq!(outer.children.len(), 1, "二层挂在一层下，不是主干下");
    assert!(!outer.children[0].resumed, "二层自己没折叠过");
    assert_eq!(outer.children[0].label, "二层");
}

/// 主干没有 resumed 一说（它永远不折叠）。
#[test]
fn trunk_is_never_resumed() {
    let session = Session::new("s1");
    trunk_turn(&session, "问一", "答一", 100);
    let tree = session.branch_tree();
    assert!(!tree.resumed);
    assert!(tree.closed.is_none());
}
