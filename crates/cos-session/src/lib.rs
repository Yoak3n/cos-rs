//! cos-session —— SessionEvent 追加日志、derive_messages、JSONL 持久化（P3）。
//!
//! 会话日志为唯一事实源（PLAN.md §0）：模型可见 ⟺ 已记录。
//! 事件为封闭枚举 + `Custom` 逃生舱（决策 D4）；`seq` 单调连续（含原始 chunk）。
//! 持久化走压缩口径（JSONL 一行一事件）：被 message 盖住的 chunk 不落，
//! 见 [`jsonl::condense_events`]——文件往返仍逐字节一致。
//!
//! 树状会话：分支归属是**事件上的事实**（`SessionEvent.branch` + `branch/open` /
//! `branch/close`），会话树只是投影（见 [`BranchNode`]）。`derive_messages` 按当前
//! 写入游标投影——于是 agent-loop 无需感知分支。
//!
//! 语义权威参考：`packages/core/session/src/types.ts`。
//! P3 范围：surface 投影（user/message、assistant/message、tool/result、Custom 透传）；
//! compaction/surface-op/seed 等高级机制留待后续阶段。

#![warn(missing_docs)]

mod derive;
mod error;
mod jsonl;
mod session;
mod types;

pub use derive::{BranchNode, branch_tree, derive_messages, visible_events};
pub use error::SessionError;
pub use jsonl::{condense_events, load_jsonl, save_jsonl};
pub use session::Session;
pub use types::{
    AbortCause, RequestContext, RequestHeader, SESSION_FORMAT_VERSION, SessionEvent,
    SessionEventData, SessionHeader, TodoItem, TodoStatus, ToolError, TurnEndReason,
};
