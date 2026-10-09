# cos-system-prompt

prompt 段装配（P5）：段带显式 `order`（装配即排序 + 查重），渲染只拼段——
工具清单不进 system，只走请求的原生 `tools` 字段。

依赖方向铁律（PLAN.md §2）：plugins/* 与 cos-agent-loop 只依赖接缝 Definition crate；
cos-core 不依赖任何上层 crate。

设计决策见 [../docs/decisions.md](../docs/decisions.md)；实施计划见 [../PLAN.md](../PLAN.md)。

