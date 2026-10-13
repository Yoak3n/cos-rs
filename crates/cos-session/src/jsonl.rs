//! JSONL 持久化：第一行 header，随后一行一事件。
//!
//! 文件是**对话记录**，不是流式过程的全量转储：落盘即压缩——被同
//! (turn, step, branch) 的 assistant/message 盖住的 chunk 不落（正文已在
//! message 里，chunk 行是纯重复）；打断轮除外（见 [`condense_events`]）。
//! 内存日志不经此口径：追加即事实源，seq 单调不动。

use std::collections::HashSet;
use std::path::Path;

use crate::error::SessionError;
use crate::session::Session;
use crate::types::{SESSION_FORMAT_VERSION, SessionEvent, SessionEventData, SessionHeader};

/// 保存会话：`header` 一行 + 每条事件一行（父目录不存在时自动创建）。
///
/// 落盘走 [`condense_events`] 压缩口径；从文件载回再存，逐字节一致（压缩对
/// 已压缩的事件集是不动点）。
pub fn save_jsonl(
    session: &Session,
    header: &SessionHeader,
    path: impl AsRef<Path>,
) -> Result<(), SessionError> {
    let events = session.events();
    let mut text = String::new();
    text.push_str(&serde_json::to_string(header)?);
    text.push('\n');
    for event in condense_events(&events) {
        text.push_str(&serde_json::to_string(&event)?);
        text.push('\n');
    }
    if let Some(parent) = path.as_ref().parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, text)?;
    Ok(())
}

/// 落盘压缩口径：滤掉被同 `(turn, step, branch)` 的 assistant/message 盖住的
/// assistant/chunk。
///
/// chunk 在流式期间逐条入账（驱动器「先写日志再行动」），message 落账时其正文
/// 已是全部 chunk 的合流——文件里两者并存是纯冗余。**打断轮不压**：流被打断时
/// message 不会入账，chunk 是该 step 部分产出的唯一记录，必须保留。
pub fn condense_events(events: &[SessionEvent]) -> Vec<SessionEvent> {
    let covered: HashSet<(u32, u32, Option<String>)> = events
        .iter()
        .filter_map(|event| match &event.data {
            SessionEventData::AssistantMessage { turn, step, .. } => {
                Some((*turn, *step, event.branch.clone()))
            }
            _ => None,
        })
        .collect();
    events
        .iter()
        .filter(|event| {
            !matches!(&event.data, SessionEventData::AssistantChunk { turn, step, .. }
                if covered.contains(&(*turn, *step, event.branch.clone())))
        })
        .cloned()
        .collect()
}

/// 载入会话：校验版本 → 逐行解析。
///
/// 返回 (header, events)；调用方可用 [`Session::from_events`] 恢复（`next_seq`
/// 取 `max(seq) + 1`，压缩留下的 seq 空洞无碍续接）。
pub fn load_jsonl(
    path: impl AsRef<Path>,
) -> Result<(SessionHeader, Vec<SessionEvent>), SessionError> {
    let text = std::fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header_line = lines
        .next()
        .ok_or_else(|| SessionError::Invalid("空日志文件".into()))?;
    let header: SessionHeader = serde_json::from_str(header_line)?;
    if header.version != SESSION_FORMAT_VERSION {
        return Err(SessionError::VersionMismatch {
            expected: SESSION_FORMAT_VERSION,
            found: header.version,
        });
    }
    let mut events = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        events.push(serde_json::from_str(line)?);
    }
    Ok((header, events))
}
