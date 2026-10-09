//! P5：prompt 段装配 + 工具 schema 收集（文本快照测试 + 段序控制）。

use cos_core::Context;
use cos_system_prompt::{PromptError, PromptSection, PromptSections, validate};
use serde_json::json;

#[test]
fn render_produces_deterministic_snapshot() {
    let root = Context::root();
    let sections = PromptSections::new(&root);
    sections
        .append(PromptSection::new("persona", 10, "你是一个助手。"))
        .unwrap();
    sections
        .append(PromptSection::new("rules", 20, "先思考再回答。"))
        .unwrap();

    let tools = vec![json!({
        "type": "function",
        "function": {
            "name": "echo",
            "description": "回声",
            "parameters": {
                "type": "object",
                "properties": { "text": { "type": "string" } },
                "required": ["text"]
            }
        }
    })];

    let rendered = sections.render(&tools);
    assert_eq!(
        rendered,
        "你是一个助手。\n\
         \n\
         先思考再回答。\n\
         \n\
         可用工具：\n\
         - echo: 回声\n\
         \x20 参数 (JSON Schema):\n\
         {\n  \"properties\": {\n    \"text\": {\n      \"type\": \"string\"\n    }\n  },\n  \"required\": [\n    \"text\"\n  ],\n  \"type\": \"object\"\n}"
    );
}

#[test]
fn render_without_tools_omits_tool_section() {
    let root = Context::root();
    let sections = PromptSections::new(&root);
    sections
        .append(PromptSection::new("persona", 10, "你好。"))
        .unwrap();
    assert_eq!(sections.render(&[]), "你好。");
}

/// 渲染顺序由 `order` 决定，与给入顺序无关。
#[test]
fn order_decides_render_order_not_input_order() {
    let root = Context::root();
    let sections = PromptSections::new(&root);
    sections
        .replace(vec![
            PromptSection::new("c", 30, "丙"),
            PromptSection::new("a", 10, "甲"),
            PromptSection::new("b", 20, "乙"),
        ])
        .unwrap();
    assert_eq!(sections.render(&[]), "甲\n\n乙\n\n丙");
    let snapshot = sections.sections();
    let names: Vec<&str> = snapshot.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["a", "b", "c"], "快照也按渲染顺序");
}

/// 往中间插一段：`append` 按 `order` 落位，不动别的段。
#[test]
fn append_lands_in_order_without_disturbing_others() {
    let root = Context::root();
    let sections = PromptSections::new(&root);
    sections
        .replace(vec![
            PromptSection::new("core", 10, "纪律"),
            PromptSection::new("stage", 30, "阶段"),
        ])
        .unwrap();
    sections
        .append(PromptSection::new("role", 20, "角色"))
        .unwrap();
    assert_eq!(sections.render(&[]), "纪律\n\n角色\n\n阶段");
}

/// 不合法一律拒收：空名 / 空文本 / 段序冲突 / 段名重复。
#[test]
fn invalid_sections_are_rejected() {
    let root = Context::root();
    let sections = PromptSections::new(&root);
    sections
        .replace(vec![PromptSection::new("core", 10, "纪律")])
        .unwrap();

    // 与现有段冲突（append 要连同现有段一起查）
    assert_eq!(
        sections.append(PromptSection::new("角色", 10, "同序")),
        Err(PromptError::OrderConflict {
            order: 10,
            first: "core".into(),
            second: "角色".into(),
        })
    );
    assert_eq!(
        sections.append(PromptSection::new("core", 20, "重名")),
        Err(PromptError::DuplicateName("core".into()))
    );
    // 单段自检
    assert_eq!(
        sections.append(PromptSection::new("  ", 20, "空名")),
        Err(PromptError::EmptyName)
    );
    assert_eq!(
        sections.append(PromptSection::new("空文本", 20, "   ")),
        Err(PromptError::EmptyText("空文本".into()))
    );
    // 拒收不留痕：段表还是原来那一段
    assert_eq!(sections.sections().len(), 1);

    // replace 整体校验
    assert!(
        validate(&[
            PromptSection::new("a", 10, "甲"),
            PromptSection::new("b", 10, "乙"),
        ])
        .is_err()
    );
    assert!(
        validate(&[
            PromptSection::new("a", 10, "甲"),
            PromptSection::new("a", 20, "乙"),
        ])
        .is_err()
    );
    assert!(validate(&[PromptSection::new("a", 10, "甲")]).is_ok());
}
