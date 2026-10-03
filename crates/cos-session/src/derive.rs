//! 会话树投影与「模型可见历史」推导。
//!
//! 两条推导都**只读事件流**（事实源），不持有状态——所以是纯函数，可直接单测。
//!
//! - [`derive_messages`]：某分支的模型可见历史 = 祖先链在分叉点之前的部分 + 本分支
//! - [`branch_tree`]：会话树投影（分叉点、嵌套关系、折叠摘要）
//!
//! 树状会话的语义（「线性对话承载树状知识」的机制级解法）：
//! 支线看得到自己从哪儿岔出来的上下文，但**看不到父分支在分岔之后的新内容**——
//! 这就是岔路物理隔离。支线完结时以摘要回流（见 [`SessionEventData::BranchClose`]）。

use cos_llm::{Message, ToolResultMessage};
use serde::{Deserialize, Serialize};

use crate::types::{SessionEvent, SessionEventData};

/// 会话树节点（**投影**，非事实源）。
///
/// 可序列化：这是给 UI / 审计用的读模型，跨进程传输是它的正常用途
/// （字段命名随会话日志的逐字段 `rename` 约定，多词字段用 camelCase）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchNode {
    /// 分支 id（`None` = 主干/根分支）。
    pub id: Option<String>,
    /// 标签（人读，如术语名）。
    pub label: String,
    /// 父分支（主干为 `None`）。
    pub parent: Option<String>,
    /// 分叉点：父分支中此 `seq` 之后开岔（主干为 0）。
    #[serde(rename = "forkSeq")]
    pub fork_seq: u64,
    /// 折叠摘要（未关闭为 `None`）。
    pub closed: Option<String>,
    /// 折叠之后又续上了（续聊 / 在其下再歇支线）——「已折叠」不等于「已完结」。
    ///
    /// 折叠只是**当时**把结论回流主干；之后这一支再动，它就是活的，
    /// 复用它的判据应是 `closed.is_none() || resumed`，而不是只看 `closed`。
    pub resumed: bool,
    /// 子分支（按开岔顺序）。
    pub children: Vec<BranchNode>,
}

impl BranchNode {
    /// 该节点（含自身）的分支总数。
    pub fn count(&self) -> usize {
        1 + self.children.iter().map(BranchNode::count).sum::<usize>()
    }

    /// 该节点是否为主干。
    pub fn is_trunk(&self) -> bool {
        self.id.is_none()
    }
}

/// 事件 → surface 消息（不参与投影的结构事件返回 `None`）。
///
/// 与 dsh `deriveMessages` 同口径：`user/message`、`assistant/message`、`tool/result`
/// 进 surface，`Custom` 原样透传（决策 D4），chunk / 边界 / 请求头 / 分支开关不参与。
fn surface(event: &SessionEvent) -> Option<Message> {
    match &event.data {
        SessionEventData::UserMessage(message) => Some(Message::User(message.clone())),
        SessionEventData::AssistantMessage { message, .. } => {
            Some(Message::Assistant(message.clone()))
        }
        SessionEventData::ToolResult {
            message, call_id, ..
        } => Some(Message::Tool(ToolResultMessage {
            content: message.content.clone(),
            // 配对调用 id 必须随历史回流（OpenAI 协议 tool 消息需要 tool_call_id）
            call_id: Some(call_id.clone()),
        })),
        SessionEventData::Custom { name, data } => Some(Message::Custom {
            name: name.clone(),
            data: data.clone(),
        }),
        _ => None,
    }
}

/// 某分支视野内的事件（祖先链在分叉点之前的部分 + 本分支自身），按祖先到自身的顺序。
///
/// 这是 [`derive_messages`] 的**取样范围**，单独导出是为了让不变量与审计复用同一份
/// 「可见」定义——树状会话下「已记录」必须理解为「本分支视野内的已记录」，
/// 支线里的追问对主干不可见是设计，不是漏记。
pub fn visible_events<'a>(
    events: &'a [SessionEvent],
    branch: Option<&str>,
) -> Vec<&'a SessionEvent> {
    let mut out = Vec::new();
    for (id, upper) in ancestry(events, branch) {
        for event in events {
            if event.branch == id && event.seq <= upper {
                out.push(event);
            }
        }
    }
    out
}

/// 某分支的模型可见历史；`branch` 为 `None` 即主干。
///
/// 非分支会话（没有任何 `branch/open`）的结果与逐事件全量投影完全一致。
pub fn derive_messages(events: &[SessionEvent], branch: Option<&str>) -> Vec<Message> {
    visible_events(events, branch)
        .into_iter()
        .filter_map(surface)
        .collect()
}

/// 祖先链（根 → 目标）及每段的可见上界 `seq`（含）。
///
/// 目标段上界为 `u64::MAX`；其余各段为其子分支的 `parent_seq`。
fn ancestry(events: &[SessionEvent], branch: Option<&str>) -> Vec<(Option<String>, u64)> {
    let mut chain = Vec::new();
    let mut current = branch.map(str::to_string);
    let mut upper = u64::MAX;
    loop {
        chain.push((current.clone(), upper));
        let Some(id) = current else { break };
        let Some((parent, fork_seq)) = events.iter().find_map(|event| match &event.data {
            SessionEventData::BranchOpen {
                branch_id,
                parent_branch,
                parent_seq,
                ..
            } if branch_id == &id => Some((parent_branch.clone(), *parent_seq)),
            _ => None,
        }) else {
            break;
        };
        upper = fork_seq;
        current = parent;
    }
    chain.reverse();
    chain
}

/// 会话树投影：主干为根，支线按开岔顺序挂上。
pub fn branch_tree(events: &[SessionEvent]) -> BranchNode {
    let mut drafts: Vec<Draft> = Vec::new();
    for event in events {
        match &event.data {
            SessionEventData::BranchOpen {
                branch_id,
                parent_branch,
                parent_seq,
                label,
            } => drafts.push(Draft {
                id: branch_id.clone(),
                parent: parent_branch.clone(),
                fork_seq: *parent_seq,
                label: label.clone(),
                closed: None,
                last_activity: *parent_seq,
                closed_at: None,
            }),
            SessionEventData::BranchClose { branch_id, summary } => {
                if let Some(draft) = drafts.iter_mut().find(|d| &d.id == branch_id) {
                    // 折叠可重复：摘要在事件流里是追加的，投影取**最新**一条
                    draft.closed = Some(summary.clone());
                    draft.closed_at = Some(event.seq);
                }
            }
            _ => {}
        }
        // 活动时间戳：事件自身归属的分支（`branch` 字段）决定它算谁的动静
        if let Some(id) = &event.branch
            && let Some(draft) = drafts.iter_mut().find(|d| &d.id == id)
        {
            draft.last_activity = draft.last_activity.max(event.seq);
        }
    }
    // 后代活动也算「续上」：折叠后又在子分支里下钻，父分支不该仍显示冻结
    for index in 0..drafts.len() {
        let mut parent = drafts[index].parent.clone();
        let activity = drafts[index].last_activity;
        while let Some(id) = parent {
            let Some(draft) = drafts.iter_mut().find(|d| d.id == id) else {
                break;
            };
            draft.last_activity = draft.last_activity.max(activity);
            parent = draft.parent.clone();
        }
    }
    BranchNode {
        id: None,
        label: "trunk".into(),
        parent: None,
        fork_seq: 0,
        closed: None,
        resumed: false,
        children: children_of(&drafts, None),
    }
}

/// 中间形态：先收齐开/关，再一次性建树（避免边扫边拼）。
struct Draft {
    id: String,
    parent: Option<String>,
    fork_seq: u64,
    label: String,
    closed: Option<String>,
    /// 本分支（含后代）最后一次活动的事件 `seq`。
    last_activity: u64,
    /// 最后一次折叠的事件 `seq`（没折叠过为 `None`）。
    closed_at: Option<u64>,
}

impl Draft {
    /// 是否「折叠后又续上」：折叠之后还有本分支或其后代的动静。
    ///
    /// 判定收在一处（不落成字段），免得与 `Session::is_closed` 的守卫各写一份口径。
    fn is_resumed(&self) -> bool {
        self.closed_at
            .is_some_and(|closed_at| self.last_activity > closed_at)
    }
}

/// 某分支折叠之后是否又有活动（续聊 / 在它下面下钻）。
///
/// [`crate::Session`] 的重复折叠守卫用它：**已折叠** ≠ **已完结**——
/// 折叠之后再说话，这一支就还是活的，应该允许再次折叠（摘要取最新）。
pub(crate) fn is_resumed(events: &[SessionEvent], branch: &str) -> bool {
    let mut drafts: Vec<Draft> = Vec::new();
    for event in events {
        match &event.data {
            SessionEventData::BranchOpen {
                branch_id,
                parent_branch,
                parent_seq,
                label,
            } => drafts.push(Draft {
                id: branch_id.clone(),
                parent: parent_branch.clone(),
                fork_seq: *parent_seq,
                label: label.clone(),
                closed: None,
                last_activity: *parent_seq,
                closed_at: None,
            }),
            SessionEventData::BranchClose { branch_id, summary } => {
                if let Some(draft) = drafts.iter_mut().find(|d| &d.id == branch_id) {
                    draft.closed = Some(summary.clone());
                    draft.closed_at = Some(event.seq);
                }
            }
            _ => {}
        }
        if let Some(id) = &event.branch
            && let Some(draft) = drafts.iter_mut().find(|d| &d.id == id)
        {
            draft.last_activity = draft.last_activity.max(event.seq);
        }
    }
    for index in 0..drafts.len() {
        let mut parent = drafts[index].parent.clone();
        let activity = drafts[index].last_activity;
        while let Some(id) = parent {
            let Some(draft) = drafts.iter_mut().find(|d| d.id == id) else {
                break;
            };
            draft.last_activity = draft.last_activity.max(activity);
            parent = draft.parent.clone();
        }
    }
    drafts
        .iter()
        .find(|d| d.id == branch)
        .is_some_and(Draft::is_resumed)
}

fn children_of(drafts: &[Draft], parent: Option<&str>) -> Vec<BranchNode> {
    drafts
        .iter()
        .filter(|d| d.parent.as_deref() == parent)
        .map(|d| BranchNode {
            id: Some(d.id.clone()),
            label: d.label.clone(),
            parent: d.parent.clone(),
            fork_seq: d.fork_seq,
            closed: d.closed.clone(),
            resumed: d.is_resumed(),
            children: children_of(drafts, Some(&d.id)),
        })
        .collect()
}
