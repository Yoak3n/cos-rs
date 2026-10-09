//! 会话日志：追加即事实源；`derive_messages` 从日志投影模型可见历史。
//!
//! P4 起 `Session` 具备内部可变性（`Arc<Mutex<Inner>>`）：Agent 句柄对外共享
//! `&Session` 只读视图，loop 驱动器以 `&self` 追加——写路径仍是单写者（loop）。
//!
//! 树状会话：`Session` 持有**写入游标** `current_branch`，`append` 给事件打分支标记。
//! 于是 agent-loop 一行都不用改——它照旧调 `derive_messages()`，拿到的自然是
//! 当前分支的上下文（见 [`crate::derive`]）。

use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use cos_llm::Message;

use crate::derive::{self, BranchNode};
use crate::error::SessionError;
use crate::types::{SessionEvent, SessionEventData};

/// 内部状态（锁保护）。
struct Inner {
    events: Vec<SessionEvent>,
    next_seq: u64,
    /// 当前写入分支（`None` = 主干）。
    current_branch: Option<String>,
    /// 下一个分支序号（生成 `br_N`）。
    next_branch: u64,
}

/// 追加式会话日志（唯一事实源：模型可见 ⟺ 已记录）。
///
/// 廉价 Clone（共享内部 `Arc`）：多个持有者看到同一份日志。
#[derive(Clone)]
pub struct Session {
    id: String,
    /// 会话创建时间（Unix epoch 毫秒）。
    ///
    /// 这是**会话自带的元数据**，不是保存方该凭空造的东西：收尾落盘（`finish_with`）
    /// 是覆盖式重写，若在这里丢了创建时间，日志头就会在第一次收尾后变成 0，
    /// 而且每次重写都再丢一次——嵌入方只能自己重读旧 header 兜回来。
    created_at_ms: u64,
    inner: Arc<Mutex<Inner>>,
}

impl Session {
    /// 新建空会话（seq 从 1 起，游标在主干，创建时间取当前）。
    pub fn new(id: impl Into<String>) -> Self {
        Self::from_events_at(id, Vec::new(), now_ms())
    }

    /// 从既有事件恢复（重载/回放）；`next_seq = max(seq) + 1`，游标回到主干。
    ///
    /// 创建时间取当前——**从日志恢复请用 [`Session::from_events_at`]**，
    /// 把原 header 的创建时间带回来，否则下一次重写就把它冲掉了。
    /// 游标回到主干（重载点不猜）：在支线里再开支线要靠显式 parent，见 [`Self::open_branch`]。
    pub fn from_events(id: impl Into<String>, events: Vec<SessionEvent>) -> Self {
        Self::from_events_at(id, events, now_ms())
    }

    /// 从既有事件 + 创建时间恢复（`load_jsonl` 读出的 header 直接喂进来）。
    pub fn from_events_at(
        id: impl Into<String>,
        events: Vec<SessionEvent>,
        created_at_ms: u64,
    ) -> Self {
        let next_seq = events.iter().map(|event| event.seq).max().unwrap_or(0) + 1;
        let next_branch = events
            .iter()
            .filter(|event| matches!(event.data, SessionEventData::BranchOpen { .. }))
            .count() as u64
            + 1;
        Self {
            id: id.into(),
            created_at_ms,
            inner: Arc::new(Mutex::new(Inner {
                events,
                next_seq,
                current_branch: None,
                next_branch,
            })),
        }
    }

    /// 会话 id。
    pub fn id(&self) -> &str {
        &self.id
    }

    /// 会话创建时间（Unix epoch 毫秒）。
    pub fn created_at_ms(&self) -> u64 {
        self.created_at_ms
    }

    /// 全部事件快照（追加顺序）。
    pub fn events(&self) -> Vec<SessionEvent> {
        self.inner.lock().unwrap().events.clone()
    }

    /// 仅返回 `seq > after` 的事件（增量读取；流式显示用，避免整表克隆）。
    pub fn events_after(&self, after: u64) -> Vec<SessionEvent> {
        let inner = self.inner.lock().unwrap();
        let start = inner.events.partition_point(|event| event.seq <= after);
        inner.events[start..].to_vec()
    }

    /// 已分配的最大 seq。
    pub fn last_seq(&self) -> u64 {
        self.inner.lock().unwrap().next_seq.saturating_sub(1)
    }

    /// 已记录的最大 turn 号（0 = 还没跑过 turn）。
    ///
    /// turn 号是**会话级**编号：驱动器换一个实例接着跑同一个会话时，必须从这里续接，
    /// 否则新实例会从 1 重新数，破坏 `turn 号连续` 不变量（进程重启、CLI 每次调用、
    /// UI 重开都属于这种情形）。
    pub fn last_turn(&self) -> u32 {
        self.inner
            .lock()
            .unwrap()
            .events
            .iter()
            .filter_map(|event| match &event.data {
                SessionEventData::TurnStart { turn } | SessionEventData::TurnEnd { turn, .. } => {
                    Some(*turn)
                }
                _ => None,
            })
            .max()
            .unwrap_or(0)
    }

    /// 追加事件（时间戳取当前 epoch 毫秒）；返回写入的事件。
    pub fn append(&self, data: SessionEventData) -> SessionEvent {
        self.append_at(data, now_ms())
    }

    /// 追加事件（显式时间戳，测试确定性用）；返回写入的事件。
    pub fn append_at(&self, data: SessionEventData, time_ms: u64) -> SessionEvent {
        let mut inner = self.inner.lock().unwrap();
        push_event(&mut inner, data, time_ms)
    }

    // ───────────────────────── 树状会话 ─────────────────────────

    /// 当前写入分支（`None` = 主干）。
    pub fn current_branch(&self) -> Option<String> {
        self.inner.lock().unwrap().current_branch.clone()
    }

    /// 从**当前分支**的 `parent_seq` 处开支线，并切进去；返回新分支 id。
    ///
    /// `parent_seq` 是「分岔点在父分支的哪个 seq 之后」——父分支视野含 `seq <= parent_seq`。
    pub fn open_branch(
        &self,
        label: impl Into<String>,
        parent_seq: u64,
    ) -> Result<String, SessionError> {
        let mut inner = self.inner.lock().unwrap();
        let last = inner.next_seq.saturating_sub(1);
        if parent_seq > last {
            return Err(SessionError::Invalid(format!(
                "分叉点 {parent_seq} 超出已记录范围（last_seq = {last}）"
            )));
        }
        let id = format!("br_{}", inner.next_branch);
        inner.next_branch += 1;
        let parent = inner.current_branch.clone();
        // 先切游标再写事件：BranchOpen 自身归属新分支（它是该分支的出生记录）
        inner.current_branch = Some(id.clone());
        push_event(
            &mut inner,
            SessionEventData::BranchOpen {
                branch_id: id.clone(),
                parent_branch: parent,
                parent_seq,
                label: label.into(),
            },
            now_ms(),
        );
        Ok(id)
    }

    /// 切换写入分支（`None` = 回主干）。
    pub fn enter_branch(&self, branch: Option<&str>) -> Result<(), SessionError> {
        let mut inner = self.inner.lock().unwrap();
        if let Some(id) = branch
            && !branch_exists(&inner.events, id)
        {
            return Err(SessionError::Invalid(format!("分支不存在: {id}")));
        }
        inner.current_branch = branch.map(str::to_string);
        Ok(())
    }

    /// 折叠**当前分支**（写摘要并关闭），游标交还父分支；返回被关闭的分支 id。
    ///
    /// 主干不能关闭（它就是会话本身）。
    ///
    /// **折叠不是终态**：已折叠的分支可以再进去说话（[`Self::enter_branch`] 不拦），
    /// 也可以在其下再开支线——那种情况下它算「续上了」（[`crate::BranchNode::resumed`]），
    /// 允许再次折叠，摘要以最新一条为准（见 [`crate::derive`]）。
    /// 真正拦住的是「折叠之后一动没动又折一次」。
    pub fn close_branch(&self, summary: impl Into<String>) -> Result<String, SessionError> {
        let mut inner = self.inner.lock().unwrap();
        let Some(id) = inner.current_branch.clone() else {
            return Err(SessionError::Invalid("主干不能折叠".into()));
        };
        if is_closed(&inner.events, &id) && !derive::is_resumed(&inner.events, &id) {
            return Err(SessionError::Invalid(format!(
                "分支已折叠且其后没有新内容: {id}"
            )));
        }
        let parent = parent_of(&inner.events, &id);
        push_event(
            &mut inner,
            SessionEventData::BranchClose {
                branch_id: id.clone(),
                summary: summary.into(),
            },
            now_ms(),
        );
        inner.current_branch = parent;
        Ok(id)
    }

    /// 指定分支的模型可见历史（**不改变游标**；审计与 UI 用）。
    pub fn derive_branch_messages(&self, branch: Option<&str>) -> Vec<Message> {
        let inner = self.inner.lock().unwrap();
        derive::derive_messages(&inner.events, branch)
    }

    /// 会话树投影（主干为根）。
    pub fn branch_tree(&self) -> BranchNode {
        derive::branch_tree(&self.inner.lock().unwrap().events)
    }

    // ───────────────────────── 投影 ─────────────────────────

    /// 从日志投影**当前分支**的模型可见历史（同 dsh `deriveMessages`）。
    pub fn derive_messages(&self) -> Vec<Message> {
        let inner = self.inner.lock().unwrap();
        derive::derive_messages(&inner.events, inner.current_branch.as_deref())
    }
}

/// 追加事件（调用方已持锁）；打上当前分支标记。
fn push_event(inner: &mut Inner, data: SessionEventData, time_ms: u64) -> SessionEvent {
    let event = SessionEvent {
        seq: inner.next_seq,
        time: time_ms,
        branch: inner.current_branch.clone(),
        data,
    };
    inner.next_seq += 1;
    inner.events.push(event.clone());
    event
}

/// 分支是否已开过。
fn branch_exists(events: &[SessionEvent], id: &str) -> bool {
    events.iter().any(|event| match &event.data {
        SessionEventData::BranchOpen { branch_id, .. } => branch_id == id,
        _ => false,
    })
}

/// 分支是否已折叠。
fn is_closed(events: &[SessionEvent], id: &str) -> bool {
    events.iter().any(|event| match &event.data {
        SessionEventData::BranchClose { branch_id, .. } => branch_id == id,
        _ => false,
    })
}

/// 分支的父分支（主干为 `None`）。
fn parent_of(events: &[SessionEvent], id: &str) -> Option<String> {
    events
        .iter()
        .find_map(|event| match &event.data {
            SessionEventData::BranchOpen {
                branch_id,
                parent_branch,
                ..
            } if branch_id == id => Some(parent_branch.clone()),
            _ => None,
        })
        .flatten()
}

/// 当前 epoch 毫秒。
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("系统时间早于 Unix epoch")
        .as_millis() as u64
}
