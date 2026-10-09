//! 会话恢复验收：`CreateAgentOptions.session` 从持久化日志续跑。
//!
//! 覆盖两条：① 事件与 seq 连续接上；② **分支游标随会话恢复**——
//! 续跑写进恢复的分支，而不是漏回主干。

use std::sync::Arc;

use cos_agent::{AgentOptions, AgentRegistry, CreateAgentOptions};
use cos_agent_loop::LoopFactory;
use cos_core::Context;
use cos_llm::{LlmAdapter, StreamChunk, UserMessage};
use cos_session::{Session, SessionEventData};
use cos_test_support::{MockAdapter, MockReply};

fn setup() -> (Context, AgentRegistry) {
    let root = Context::root();
    let registry = AgentRegistry::new(&root);
    root.provide(registry.clone()).unwrap();
    registry.set_factory(Arc::new(LoopFactory)).unwrap();
    (root, registry)
}

fn adapter(replies: Vec<&str>) -> Arc<dyn LlmAdapter> {
    Arc::new(MockAdapter::new(
        "mock",
        replies
            .into_iter()
            .map(|text| MockReply::new(vec![StreamChunk::text(text)]))
            .collect(),
    ))
}

fn options(
    session_id: &str,
    session: Option<Session>,
    adapter: Arc<dyn LlmAdapter>,
) -> CreateAgentOptions {
    CreateAgentOptions {
        session,
        session_id: session_id.into(),
        options: AgentOptions {
            provider: Some("mock".into()),
            model: Some("mock".into()),
            max_tokens: None,
            role: None,
        },
        adapter,
    }
}

/// seq 从 1 起连续无空洞。
fn assert_seq_contiguous(session: &Session) {
    let seqs: Vec<u64> = session.events().iter().map(|e| e.seq).collect();
    assert_eq!(
        seqs,
        (1..=seqs.len() as u64).collect::<Vec<u64>>(),
        "恢复后 seq 必须连续"
    );
}

#[tokio::test]
async fn agent_resumes_from_a_persisted_session() {
    let (_root, registry) = setup();

    let first = registry
        .create(options("resume-1", None, adapter(vec!["第一轮答"])))
        .await
        .unwrap();
    first.followup(UserMessage::new("第一轮问"));
    first.when_idle().await;
    let events = first.session().events();
    let baseline = events.len();
    assert!(baseline > 0);
    assert_seq_contiguous(first.session());

    // 从日志恢复：**换一个装配上下文**（session_id 是注册表的键，同 id 不能并存；
    // 真实的「恢复」发生在进程重启后），新 agent 带着既有事件而不是从零开始
    let (_root2, registry2) = setup();
    let resumed = registry2
        .create(options(
            "resume-1",
            Some(Session::from_events("resume-1", events.clone())),
            adapter(vec!["第二轮答"]),
        ))
        .await
        .unwrap();
    assert_eq!(resumed.session().events(), events, "恢复后应看到既有事件");

    resumed.followup(UserMessage::new("第二轮问"));
    resumed.when_idle().await;
    let after = resumed.session().events();
    assert!(after.len() > baseline, "续跑应追加事件");
    assert_eq!(&after[..baseline], &events[..], "既有事件不被改写");
    assert_seq_contiguous(resumed.session());

    // 第二轮的问与答都在
    let texts: Vec<String> = after
        .iter()
        .filter_map(|e| match &e.data {
            SessionEventData::UserMessage(m) => Some(m.content.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(texts, ["第一轮问", "第二轮问"]);
}

#[tokio::test]
async fn branch_cursor_survives_resume() {
    let session = Session::new("br-resume");
    session.append(SessionEventData::UserMessage(UserMessage::new("主干问"))); // seq 1
    session.open_branch("支线", 1).unwrap(); // seq 2，归属 br_1

    let restored = Session::from_events("br-resume", session.events());
    // `from_events` 把游标放回主干（恢复点不猜），调用方显式进分支
    assert!(restored.current_branch().is_none());
    restored.enter_branch(Some("br_1")).unwrap();
    assert_eq!(restored.current_branch().as_deref(), Some("br_1"));

    // 恢复后新开分支的 id 不与既有的撞车
    restored.enter_branch(None).unwrap();
    assert_eq!(
        restored.open_branch("新支线", restored.last_seq()).unwrap(),
        "br_2"
    );
}

#[tokio::test]
async fn agent_writes_into_the_restored_branch() {
    let session = Session::new("br-agent");
    session.append(SessionEventData::UserMessage(UserMessage::new("主干问"))); // seq 1
    session.open_branch("支线", 1).unwrap(); // seq 2
    let baseline = session.last_seq();

    let restored = Session::from_events("br-agent", session.events());
    restored.enter_branch(Some("br_1")).unwrap();

    let (_root, registry) = setup();
    let agent = registry
        .create(options("br-agent", Some(restored), adapter(vec!["支线答"])))
        .await
        .unwrap();
    agent.followup(UserMessage::new("支线追问"));
    agent.when_idle().await;

    let events = agent.session().events();
    let fresh: Vec<_> = events.iter().filter(|e| e.seq > baseline).collect();
    assert!(!fresh.is_empty(), "应有新事件");
    assert!(
        fresh.iter().all(|e| e.branch.as_deref() == Some("br_1")),
        "续跑必须写进恢复的分支，实际 {:?}",
        fresh.iter().map(|e| e.branch.clone()).collect::<Vec<_>>()
    );
    // 支线视野含主干在分叉点之前的上下文
    let view = agent.session().derive_branch_messages(Some("br_1"));
    let has_trunk = view
        .iter()
        .any(|m| matches!(m, cos_llm::Message::User(u) if u.content == "主干问"));
    assert!(has_trunk, "支线应看得到分叉点之前的祖先上下文");
}

/// turn 号是**会话级**编号：换一个 agent 实例续跑，turn 从日志续接而不是重新从 1 数。
///
/// 回归点：驱动器把 `last_turn` 存在自己身上（构造时为 0），于是「每次调用都新建 agent」
/// 的用法（CLI 每次执行、进程重启、UI 重开）会把每一轮都写成 turn 1，
/// 破坏 `turn 号连续` 不变量（`cos-invariants` 的 `turn-pairing`）。
#[tokio::test]
async fn turn_numbering_continues_across_agent_instances() {
    let (_root, registry) = setup();
    let first = registry
        .create(options("turn-1", None, adapter(vec!["第一轮答"])))
        .await
        .unwrap();
    first.followup(UserMessage::new("第一轮问"));
    first.when_idle().await;
    assert_eq!(first.session().last_turn(), 1, "首轮应为 turn 1");
    let events = first.session().events();

    let (_root2, registry2) = setup();
    let resumed = registry2
        .create(options(
            "turn-1",
            Some(Session::from_events("turn-1", events)),
            adapter(vec!["第二轮答"]),
        ))
        .await
        .unwrap();
    resumed.followup(UserMessage::new("第二轮问"));
    resumed.when_idle().await;

    let turns: Vec<u32> = resumed
        .session()
        .events()
        .iter()
        .filter_map(|event| match &event.data {
            SessionEventData::TurnStart { turn } => Some(*turn),
            _ => None,
        })
        .collect();
    assert_eq!(turns, [1, 2], "turn 号必须跨实例连续");
    assert_eq!(resumed.session().last_turn(), 2);
}
