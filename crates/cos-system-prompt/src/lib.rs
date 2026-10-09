//! cos-system-prompt —— prompt 段装配（P5）。
//!
//! 语义参考：`packages/core/system-prompt/src/`（P5 简化：有序段列表，
//! 变量/条件段等高级机制留待后续阶段；渲染文本确定性、可快照）。
//!
//! **不写工具清单**：工具以原生 `tools` 字段随请求发出（适配器各自转换格式），
//! 再在 system 里抄一遍既费 token，又可能与原生描述打架——所以 [`PromptSections::render`]
//! 只拼段。
//!
//! **段序是显式的**：每段自带 [`PromptSection::order`]，装配按它升序存放——
//! 顺序不靠 `append` 的调用次序（那是隐式的：往中间加一段就会挪动后面所有段）。
//! 段名与段序都不许重复、名与文本不许为空（[`PromptError`]），所以
//! 「这次 system 由哪些段、按什么序拼成」随时可查（[`PromptSections::sections`]）。

#![warn(missing_docs)]

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use cos_core::{Context, Service};
use thiserror::Error;

/// 一段 prompt。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptSection {
    /// 段名（如 "persona"、"rules"；用于日志与追踪，不参与渲染文本）。
    pub name: String,
    /// 段序：装配按它升序排列。**显式给**——建议 10 一档，往中间插段不必动别段；
    /// 负数表示"排在最前"。
    pub order: i32,
    /// 段文本。
    pub text: String,
}

impl PromptSection {
    /// 建一段（`order` 显式）。
    pub fn new(name: impl Into<String>, order: i32, text: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            order,
            text: text.into(),
        }
    }
}

/// 段装配被拒的原因：不合法就进不去，「可控」从这里开始。
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PromptError {
    /// 段名为空。
    #[error("提示词段名不能为空")]
    EmptyName,
    /// 段文本为空。
    #[error("提示词段 {0} 的文本为空")]
    EmptyText(String),
    /// 段序冲突（同序两段的先后无法判定）。
    #[error("提示词段序冲突：order={order} 被 {first} / {second} 占用")]
    OrderConflict {
        /// 冲突的段序。
        order: i32,
        /// 先占用的段名。
        first: String,
        /// 后到的段名。
        second: String,
    },
    /// 段名重复。
    #[error("提示词段名重复：{0}")]
    DuplicateName(String),
}

/// prompt 装配服务（`ctx.provide` 为 `"system-prompt"`）。
pub struct PromptSections {
    /// 始终按 `order` 升序存放（装配时排好，渲染只做拼接）。
    sections: Mutex<Vec<PromptSection>>,
}

impl Service for PromptSections {
    const NAME: &'static str = "system-prompt";
}

impl PromptSections {
    /// 空装配器。
    pub fn new(_root: &Context) -> Self {
        Self {
            sections: Mutex::new(Vec::new()),
        }
    }

    /// 追加一段（按其 `order` 插入到正确位置；段名或段序与现有段冲突即拒）。
    pub fn append(&self, section: PromptSection) -> Result<(), PromptError> {
        let mut sections = self.sections.lock().unwrap();
        let mut merged = sections.clone();
        merged.push(section);
        validate(&merged)?;
        merged.sort_by_key(|section| section.order);
        *sections = merged;
        Ok(())
    }

    /// 整体替换段列表（= 应用一套新的「段 + order」编排；存储即按 `order` 排好）。
    pub fn replace(&self, sections: Vec<PromptSection>) -> Result<(), PromptError> {
        validate(&sections)?;
        let mut sorted = sections;
        sorted.sort_by_key(|section| section.order);
        *self.sections.lock().unwrap() = sorted;
        Ok(())
    }

    /// 当前段快照（**按渲染顺序**，即 `order` 升序）。
    pub fn sections(&self) -> Vec<PromptSection> {
        self.sections.lock().unwrap().clone()
    }

    /// 渲染完整 system prompt：各段按 `order` 升序、以空行分隔。
    ///
    /// 确定性输出（可快照测试）。**不含工具清单**——工具只走请求的原生 `tools`
    /// 字段（见模块文档）。
    pub fn render(&self) -> String {
        self.sections()
            .into_iter()
            .map(|section| section.text)
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

/// 段集合自检：名 / 文本非空、段名唯一、段序唯一。
pub fn validate(sections: &[PromptSection]) -> Result<(), PromptError> {
    let mut names: HashSet<&str> = HashSet::new();
    let mut orders: HashMap<i32, &str> = HashMap::new();
    for section in sections {
        if section.name.trim().is_empty() {
            return Err(PromptError::EmptyName);
        }
        if section.text.trim().is_empty() {
            return Err(PromptError::EmptyText(section.name.clone()));
        }
        if !names.insert(section.name.as_str()) {
            return Err(PromptError::DuplicateName(section.name.clone()));
        }
        if let Some(first) = orders.insert(section.order, section.name.as_str()) {
            return Err(PromptError::OrderConflict {
                order: section.order,
                first: first.to_string(),
                second: section.name.clone(),
            });
        }
    }
    Ok(())
}
