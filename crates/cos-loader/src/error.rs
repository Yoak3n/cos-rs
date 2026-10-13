//! 装载边界错误（决策 D5）：带插件/条目上下文的可读报错（同 JS `failed to apply loader entry`）。

use cos_core::CoreError;
use thiserror::Error;

/// cos-loader 的边界错误。
#[derive(Debug, Error)]
pub enum LoadError {
    /// 读取配置失败。
    #[error("读取配置失败: {0}")]
    Io(#[from] std::io::Error),
    /// 解析 cordis.yml 失败。
    #[error("解析 cordis.yml 失败: {0}")]
    Yaml(#[from] serde_saphyr::Error),
    /// 插件工厂未注册（静态注册表里没有这个名字）。
    #[error("插件 '{name}' 未注册（可用: {available:?}）")]
    UnknownPlugin {
        /// 配置里的工厂名。
        name: String,
        /// 已注册的工厂名（便于排查拼写）。
        available: Vec<&'static str>,
    },
    /// 条目配置反序列化失败。
    #[error("failed to apply loader entry {id} ({name}): 配置解析失败: {source}")]
    ConfigParse {
        /// 条目实例 id。
        id: String,
        /// 工厂名。
        name: String,
        /// serde 反序列化错误。
        #[source]
        source: serde_json::Error,
    },
    /// 条目配置校验失败（`Validate::validate`）。
    #[error("failed to apply loader entry {id} ({name}): 配置校验失败: {source}")]
    ConfigInvalid {
        /// 条目实例 id。
        id: String,
        /// 工厂名。
        name: String,
        /// 校验错误。
        #[source]
        source: CoreError,
    },
    /// 依赖服务无提供者。
    #[error("服务 '{service}' 无提供者（'{plugin}' 需要）")]
    MissingDependency {
        /// 需要该服务的插件条目。
        plugin: String,
        /// 缺失的服务名。
        service: String,
    },
    /// 同一服务被多个插件提供。
    #[error("服务 '{service}' 被重复提供: {plugins:?}")]
    DuplicateProvide {
        /// 冲突的服务名。
        service: String,
        /// 提供者条目 id 列表。
        plugins: Vec<String>,
    },
    /// 依赖环（无法拓扑排序的插件条目）。
    #[error("依赖环: {cycle:?}（无法确定装载顺序）")]
    DependencyCycle {
        /// 环上的插件条目 id。
        cycle: Vec<String>,
    },
    /// 插件 apply 失败。
    #[error("failed to apply loader entry {id} ({name}): {source}")]
    Apply {
        /// 条目实例 id。
        id: String,
        /// 工厂名。
        name: String,
        /// apply 返回的错误。
        #[source]
        source: CoreError,
    },
    /// dlopen 插件：库文件打开失败。
    #[error("dlopen 插件 '{name}' 打开失败（{path}）: {detail}")]
    DlopenOpen {
        /// 工厂名。
        name: String,
        /// 库文件路径。
        path: String,
        /// libloading 错误。
        detail: String,
    },
    /// dlopen 插件：导出符号缺失。
    #[error("dlopen 插件 '{name}' 缺少导出符号 '{symbol}': {detail}")]
    DlopenSymbol {
        /// 工厂名。
        name: String,
        /// 缺失的符号。
        symbol: String,
        /// libloading 错误。
        detail: String,
    },
    /// dlopen 插件：B-ABI 版本不匹配（fail loud）。
    #[error("dlopen 插件 '{name}' ABI 版本不兼容（宿主 {host:?}, 插件 {plugin:?}）")]
    AbiMismatch {
        /// 工厂名。
        name: String,
        /// 宿主 B-ABI 版本。
        host: cos_contract::ContractVersion,
        /// 插件声明的 B-ABI 版本。
        plugin: cos_contract::ContractVersion,
    },
    /// dlopen 插件：apply 返回错误码。
    #[error("dlopen 插件 '{name}' apply 失败（code={code}）: {message}")]
    DlopenApply {
        /// 工厂名。
        name: String,
        /// 插件返回的错误码。
        code: i32,
        /// 插件写入的错误文本。
        message: String,
    },
    /// dlopen 插件：配置预校验（`cos_plugin_validate`）返回错误码（P12）。
    #[error("dlopen 插件 '{name}' 配置校验失败（code={code}）: {message}")]
    DlopenValidate {
        /// 工厂名。
        name: String,
        /// 插件返回的错误码。
        code: i32,
        /// 插件写入的错误文本。
        message: String,
    },
    /// patch 文件读取失败（P13）。
    #[error("patch 文件 {path} 读取失败: {source}")]
    PatchFile {
        /// patch 文件路径。
        path: String,
        /// 读取错误。
        #[source]
        source: std::io::Error,
    },
    /// patch 结构无效（P13；fail loud——dsh warn+skip 是热重载妥协，cos 静态装载直接报错）。
    #[error("{0}")]
    PatchInvalid(String),
    /// patch 定位不到目标条目（P13）。
    #[error("patch 条目 '{target}' 在 '{file}' 中找不到（现有条目: {available:?}）")]
    PatchTargetMissing {
        /// patch 声明的定位键。
        target: String,
        /// patch 文件来源。
        file: String,
        /// 当前条目 id 列表（便于排查拼写）。
        available: Vec<String>,
    },
    /// patch 名称校验失败（P13）。
    #[error("patch 条目 '{id}' 名称不匹配（patch 声明 {expected:?}，实际 {actual:?}）")]
    PatchNameMismatch {
        /// patch 定位键。
        id: String,
        /// patch 声明的名称。
        expected: String,
        /// 目标条目实际名称。
        actual: String,
    },
    /// 其他失败。
    #[error("{0}")]
    Other(String),
}
