//! cordis.yml 条目解析与 patch 层叠（语义参考：`vendor/loader/src/config/entry.ts`
//! 的 `EntryOptions` 与 `vendor/include/src/index.ts` 的 `applyEntryPatches`）。
//!
//! P13 层叠（对标 dsh 的 cordis.patch.yml 体系，静态装载变体）：
//! - 主 `cordis.yml`：顶层数组（纯条目，向后兼容）或对象 `{ patch: [..], entries: [..] }`；
//! - patch 文件 = 顶层数组，按 id/name 定位覆盖、`disabled` 禁用、`insert` 追加
//!   （无 id = 列表尾追加；带 id = 不支持——v1 无 group，fail loud）；
//! - 应用顺序：主 yml 条目 → 主 yml `patch:` 声明的文件（按序）→ 同目录
//!   `cordis.patch.yml`（自动应用，若未被显式声明）→ CLI `--patch`（按 argv 顺序）；
//!   后覆盖先。
//! - **fail loud**：patch 定位不到目标 / 名称不匹配 / insert 带 id → 启动失败
//!   （dsh warn+skip 是热重载的妥协；cos 静态装载没有这个理由，见 decisions.md P13）。

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::LoadError;

/// 一个已配置的插件节点（v1：cordis.yml 单一清单；P13 起可被 patch 覆盖/追加）。
#[derive(Debug, Clone, Deserialize)]
pub struct Entry {
    /// 稳定实例 id（缺省 = `name`）。
    #[serde(default)]
    pub id: Option<String>,
    /// 工厂名（[`crate::resolve_factory`] 的键）。
    pub name: String,
    /// 插件配置（YAML 原位解析为 JSON；缺省 `null`）。
    #[serde(default)]
    pub config: serde_json::Value,
    /// 额外依赖的服务名（叠加在插件自身 `inject()` 之上）。
    #[serde(default)]
    pub inject: Vec<String>,
    /// 禁用：跳过装载（同 dsh 的 disabled 字段）。
    #[serde(default)]
    pub disabled: bool,
    /// 条目来源（`cordis.yml` / patch 文件路径；`--dump-config` 输出）。
    /// 反序列化跳过——组装期由 [`Profile`] 标注。
    #[serde(skip)]
    pub source: String,
}

impl Entry {
    /// 有效实例 id（缺省 = 工厂名）。
    pub fn id(&self) -> &str {
        self.id.as_deref().unwrap_or(&self.name)
    }
}

/// cordis.patch.yml 条目（patch 语义，对标 dsh `applyEntryPatches`；fail loud 变体）。
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Patch {
    /// 定位键（缺省 = `name`；两者皆无且无 insert → 无效，fail loud）。
    pub id: Option<String>,
    /// 名称校验（若给出且与目标条目 `name` 不匹配 → fail loud；不覆盖名称——
    /// 同 dsh：name 只校验不进 overrides）。
    pub name: Option<String>,
    /// 追加条目（**无 id** = 列表尾追加；带 id = v1 无 group，不支持 → fail loud）。
    pub insert: Option<Vec<Entry>>,
    /// 覆盖目标条目配置（整体替换，同 dsh `target[key] = value`）。
    pub config: Option<serde_json::Value>,
    /// 覆盖目标条目 inject（整体替换）。
    pub inject: Option<Vec<String>>,
    /// 覆盖目标条目 disabled。
    pub disabled: Option<bool>,
}

/// cordis.yml：顶层 YAML 数组（v1）或 `{ patch?: [..], entries?: [..] }` 对象（P13）。
#[derive(Debug, Clone, Default)]
pub struct Profile {
    /// 合并后的条目列表（主 yml 条目 + 各 patch 层）。
    pub entries: Vec<Entry>,
    /// 主 yml 顶层 `patch:` 声明的 patch 文件（相对主 yml 目录；未合并前）。
    pub patch_files: Vec<String>,
}

impl<'de> Deserialize<'de> for Profile {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error as _;
        // 中间值走 serde_json::Value（本 crate 已有依赖；条目 `config` 本来就是
        // 「YAML 原位解析为 JSON」的口径）。顶层要么是数组（v1 纯条目），要么是
        // 对象 `{ patch, entries }`；对象里的未知字段 fail loud。
        let value = serde_json::Value::deserialize(deserializer)?;
        match value {
            serde_json::Value::Array(entries) => Ok(Profile {
                entries: entries
                    .into_iter()
                    .map(serde_json::from_value)
                    .collect::<Result<_, _>>()
                    .map_err(D::Error::custom)?,
                patch_files: Vec::new(),
            }),
            serde_json::Value::Object(mapping) => {
                let mut entries = Vec::new();
                let mut patch_files = Vec::new();
                for (key, value) in mapping {
                    match key.as_str() {
                        "entries" => {
                            entries = serde_json::from_value(value).map_err(D::Error::custom)?
                        }
                        "patch" => {
                            patch_files = serde_json::from_value(value).map_err(D::Error::custom)?
                        }
                        other => {
                            return Err(D::Error::custom(format!(
                                "cordis.yml 未知顶层字段: {other:?}（支持 entries / patch）"
                            )));
                        }
                    }
                }
                Ok(Profile {
                    entries,
                    patch_files,
                })
            }
            _ => Err(D::Error::custom(
                "cordis.yml 必须是顶层数组（- name: ...）或对象（{ patch, entries }）",
            )),
        }
    }
}

impl Profile {
    /// 解析 YAML 文本（主 cordis.yml；条目来源标注为 `cordis.yml`）。
    pub fn parse(yaml: &str) -> Result<Self, LoadError> {
        let mut profile: Profile = serde_saphyr::from_str(yaml)?;
        for entry in &mut profile.entries {
            entry.source = "cordis.yml".into();
        }
        Ok(profile)
    }

    /// 解析 patch 文件 YAML 文本（顶层数组；不是数组 → fail loud）。
    pub fn parse_patch(yaml: &str, source: &str) -> Result<Vec<Patch>, LoadError> {
        let value: serde_json::Value = serde_saphyr::from_str(yaml)
            .map_err(|error| LoadError::PatchInvalid(format!("'{source}' 解析失败: {error}")))?;
        let entries = value.as_array().ok_or_else(|| {
            LoadError::PatchInvalid(format!("'{source}' 必须是顶层数组（patch 条目列表）"))
        })?;
        entries
            .iter()
            .map(|entry| serde_json::from_value(entry.clone()))
            .map(|result| {
                result.map_err(|error| {
                    LoadError::PatchInvalid(format!("'{source}' 条目解析失败: {error}"))
                })
            })
            .collect()
    }

    /// 读取并解析 YAML 文件（主 cordis.yml；不做 patch 合并）。
    pub fn load(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        let text = std::fs::read_to_string(path)?;
        Self::parse(&text)
    }

    /// 完整装载路径：主 yml（含顶层 `patch:` 声明）→ 同目录 `cordis.patch.yml`
    /// （自动应用，若未被显式声明）→ CLI `--patch` 文件，合并为完整条目列表。
    /// `--dump-config` 与装载共用此路径（输出 = 装载）。
    ///
    /// patch 文件路径解析：主 yml `patch:` 相对主 yml 所在目录；CLI `--patch`
    /// 相对当前工作目录。
    pub fn load_merged(path: impl AsRef<Path>, cli_patches: &[String]) -> Result<Self, LoadError> {
        let path = path.as_ref();
        let mut profile = Self::load(path)?;
        let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
        // 同目录 cordis.patch.yml 自动应用（除非已被主 yml 显式声明——去重防双应用）
        let auto_patch = base_dir.join("cordis.patch.yml");
        let mut patch_paths: Vec<PathBuf> = profile
            .patch_files
            .iter()
            .map(|file| base_dir.join(file))
            .collect();
        if auto_patch.exists() && !patch_paths.iter().any(|p| same_file(p, &auto_patch)) {
            patch_paths.push(auto_patch);
        }
        for file in cli_patches {
            patch_paths.push(PathBuf::from(file));
        }
        for patch_path in patch_paths {
            let text =
                std::fs::read_to_string(&patch_path).map_err(|source| LoadError::PatchFile {
                    path: patch_path.display().to_string(),
                    source,
                })?;
            let patches = Self::parse_patch(&text, &patch_path.display().to_string())?;
            for patch in &patches {
                profile.apply_patch(patch, &patch_path.display().to_string())?;
            }
        }
        Ok(profile)
    }

    /// 应用一条 patch（fail loud）：
    /// - `insert`（无 id）→ 追加到列表尾；
    /// - 按 id/name 定位已有条目 → 名称校验 → config/inject/disabled 覆盖。
    fn apply_patch(&mut self, patch: &Patch, source: &str) -> Result<(), LoadError> {
        if let Some(inserts) = &patch.insert {
            if patch.id.is_some() || patch.name.is_some() {
                return Err(LoadError::PatchInvalid(format!(
                    "'{source}' 的 insert 不能带 id/name（v1 无 group，insert 只能追加到列表尾）"
                )));
            }
            for mut entry in inserts.clone() {
                entry.source = source.to_string();
                self.entries.push(entry);
            }
            return Ok(());
        }
        let target = patch
            .id
            .as_deref()
            .or(patch.name.as_deref())
            .ok_or_else(|| {
                LoadError::PatchInvalid(format!("'{source}' 的条目既无 id/name 也无 insert"))
            })?;
        let index = self
            .entries
            .iter()
            .position(|entry| entry.id() == target)
            .ok_or_else(|| LoadError::PatchTargetMissing {
                target: target.to_string(),
                file: source.to_string(),
                available: self
                    .entries
                    .iter()
                    .map(|entry| entry.id().to_string())
                    .collect(),
            })?;
        if let Some(name) = &patch.name
            && name != &self.entries[index].name
        {
            return Err(LoadError::PatchNameMismatch {
                id: target.to_string(),
                expected: name.clone(),
                actual: self.entries[index].name.clone(),
            });
        }
        let entry = &mut self.entries[index];
        if let Some(config) = &patch.config {
            entry.config = config.clone();
        }
        if let Some(inject) = &patch.inject {
            entry.inject = inject.clone();
        }
        if let Some(disabled) = patch.disabled {
            entry.disabled = disabled;
        }
        Ok(())
    }
}

/// 两路径是否指向同一文件（存在时按规范化路径比较；否则按词法比较）。
fn same_file(left: &Path, right: &Path) -> bool {
    match (left.canonicalize(), right.canonicalize()) {
        (Ok(left), Ok(right)) => left == right,
        _ => left == right,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(yaml: &str) -> Profile {
        Profile::parse(yaml).unwrap()
    }

    fn patch(yaml: &str) -> Patch {
        serde_saphyr::from_str(yaml).unwrap()
    }

    /// 主 yml 双形态：数组（v1 兼容）与对象 `{ patch, entries }`。
    #[test]
    fn main_yml_array_and_object_forms() {
        let array = profile("- name: todo\n- id: m1\n  name: llm\n");
        assert_eq!(array.entries.len(), 2);
        assert!(array.patch_files.is_empty());
        assert_eq!(array.entries[0].source, "cordis.yml");

        let object = profile("patch: [third-party/cordis.patch.yml]\nentries:\n- name: todo\n");
        assert_eq!(object.entries.len(), 1);
        assert_eq!(object.patch_files, vec!["third-party/cordis.patch.yml"]);

        // 未知顶层字段 → fail loud
        let error = Profile::parse("nope: 1\nentries: []\n")
            .unwrap_err()
            .to_string();
        assert!(error.contains("未知顶层字段"), "{error}");
        // 顶层标量 → fail loud
        let error = Profile::parse("just-a-string\n").unwrap_err().to_string();
        assert!(error.contains("顶层数组"), "{error}");
    }

    /// patch 按 id 覆盖 config / inject / disabled；条目来源标注为 patch 文件。
    #[test]
    fn patch_overrides_config_inject_disabled() {
        let mut base = profile("- id: main\n  name: llm\n  config: { model: a }\n- name: todo\n");
        base.apply_patch(
            &patch("id: main\nconfig: { model: b }\ninject: [svc]\ndisabled: true\n"),
            "cordis.patch.yml",
        )
        .unwrap();
        assert_eq!(base.entries[0].config["model"], "b");
        assert_eq!(base.entries[0].inject, vec!["svc"]);
        assert!(base.entries[0].disabled);
        // 未命中的条目不受影响
        assert_eq!(base.entries[1].name, "todo");
        assert!(!base.entries[1].disabled);
    }

    /// insert（无 id）追加到列表尾，来源标注为 patch 文件（第三方 B 插件入口形态）。
    #[test]
    fn patch_insert_appends_with_source() {
        let mut base = profile("- name: todo\n");
        base.apply_patch(
            &patch("insert:\n- name: ./plugins/third-party/plugin.dll\n  config: { key: v }\n"),
            "third-party/cordis.patch.yml",
        )
        .unwrap();
        assert_eq!(base.entries.len(), 2);
        assert_eq!(base.entries[1].name, "./plugins/third-party/plugin.dll");
        assert_eq!(base.entries[1].source, "third-party/cordis.patch.yml");
        assert_eq!(base.entries[1].config["key"], "v");
        // insert 带 id → fail loud（v1 无 group）
        let mut base = profile("- name: todo\n");
        let error = base
            .apply_patch(&patch("id: todo\ninsert: []\n"), "p.yml")
            .unwrap_err()
            .to_string();
        assert!(error.contains("insert 不能带 id"), "{error}");
    }

    /// 定位不到目标 → fail loud（列出可用条目）；name 不匹配 → fail loud。
    #[test]
    fn patch_missing_target_and_name_mismatch_fail_loud() {
        let mut base = profile("- name: todo\n- name: llm\n");
        let error = base
            .apply_patch(&patch("id: nope\nconfig: {}\n"), "p.yml")
            .unwrap_err()
            .to_string();
        assert!(error.contains("'nope'"), "{error}");
        assert!(error.contains("p.yml"), "{error}");
        assert!(error.contains("todo"), "{error}");
        assert!(error.contains("llm"), "{error}");
        // 只给 name（无 id）→ 以 name 定位
        let mut base = profile("- name: todo\n");
        base.apply_patch(&patch("name: todo\ndisabled: true\n"), "p.yml")
            .unwrap();
        assert!(base.entries[0].disabled);
        // 无 id 无 name 无 insert → fail loud
        let mut base = profile("- name: todo\n");
        let error = base
            .apply_patch(&patch("config: {}\n"), "p.yml")
            .unwrap_err()
            .to_string();
        assert!(error.contains("既无 id/name 也无 insert"), "{error}");
        // name 校验失败 → fail loud（不覆盖名称）
        let mut base = profile("- name: todo\n");
        let error = base
            .apply_patch(&patch("id: todo\nname: llm\n"), "p.yml")
            .unwrap_err()
            .to_string();
        assert!(error.contains("名称不匹配"), "{error}");
        assert_eq!(base.entries[0].name, "todo");
    }

    /// 后 patch 可定位先 patch insert 的条目（同层内增量索引，对标 dsh）。
    #[test]
    fn later_patch_can_target_inserted_entry() {
        let mut base = profile("- name: todo\n");
        base.apply_patch(
            &patch("insert:\n- id: tp\n  name: ./plugins/tp.dll\n"),
            "p1.yml",
        )
        .unwrap();
        base.apply_patch(&patch("id: tp\nconfig: { key: v }\n"), "p2.yml")
            .unwrap();
        assert_eq!(base.entries[1].config["key"], "v");
    }

    /// patch 文件解析：非法 YAML / 非数组 → fail loud。
    #[test]
    fn parse_patch_rejects_bad_shape() {
        assert!(Profile::parse_patch("id: 1\n", "p.yml").is_err());
        let error = Profile::parse_patch("just: {a: 1}\n", "p.yml")
            .unwrap_err()
            .to_string();
        assert!(error.contains("顶层数组"), "{error}");
        let error = Profile::parse_patch("- id: [bad\n", "p.yml")
            .unwrap_err()
            .to_string();
        assert!(error.contains("p.yml"), "{error}");
    }
}
