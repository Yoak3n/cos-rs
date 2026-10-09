//! P5：prompt 段装配（文本快照 + 段序控制；**不含工具清单**）。

use cos_core::Context;
use cos_system_prompt::{PromptError, PromptSection, PromptSections, validate};

#[test]
fn render_joins_sections_in_order() {
    let root = Context::root();
    let sections = PromptSections::new(&root);
    sections
        .append(PromptSection::new("persona", 10, "你是一个助手。"))
        .unwrap();
    sections
        .append(PromptSection::new("rules", 20, "先思考再回答。"))
        .unwrap();

    assert_eq!(
        sections.render(),
        "你是一个助手。\n\n先思考再回答。",
        "渲染 = 各段文本按 order 升序、空行分隔（确定性、可快照）"
    );
}

/// 回归点：system 里**不再**出现工具清单——工具只走请求的原生 `tools` 字段。
///
/// 曾经 `render(&tools)` 会把「名字 + 描述 + pretty 打印的完整 JSON Schema」抄进
/// system，与原生字段重复（每步都发，且两处描述不一致时打架）。
#[test]
fn render_never_lists_tools() {
    let root = Context::root();
    let sections = PromptSections::new(&root);
    sections
        .append(PromptSection::new("persona", 10, "你好。"))
        .unwrap();

    let rendered = sections.render();
    assert_eq!(rendered, "你好。");
    assert!(
        !rendered.contains("可用工具") && !rendered.contains("JSON Schema"),
        "system 不该再抄工具清单：{rendered}"
    );
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
    assert_eq!(sections.render(), "甲\n\n乙\n\n丙");
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
    assert_eq!(sections.render(), "纪律\n\n角色\n\n阶段");
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
