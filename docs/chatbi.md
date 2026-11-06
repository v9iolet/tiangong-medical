# ChatBI 当前代码说明

当前实现位于 `src/agents/nl2sql`，以本次提交的代码为准，不沿用上一版实现的测试结论。

## 文件职责

- `app.py`：独立 Gradio 页面，包含 `demo` 及直接启动入口。
- `engine.py`：SQL 生成、执行、摘要及对话上下文。
- `prompts.py`：集中维护提示词。
- `security.py`：SQL 检查和角色过滤。
- `chart_advisor.py`：图表推荐与 Plotly 渲染。
- `echarts_builder.py`：ECharts 配置生成。
- `router.py`：API 路由。

## 提交与验证范围

按要求未运行测试，未验证数据库、模型调用、页面启动或权限安全性。未将 ChatBI 注册到 `src/main.py`。

代码阅读发现，部分模块仍从 `src.nl2sql` 导入，而当前实际目录为 `src/agents/nl2sql`，启动前需修正导入路径。本次保留用户修改，不额外修复功能。旧的 `scripts/gradio_chatbi_app.py` 依赖已不存在的 `create_app` 工厂，因此与旧版测试、验证报告及测试数据脚本一起清理，不再作为启动方式。

共享配置仍保留在项目配置体系中；旧版 `CHATBI_*` 配置不代表当前代码已接入对应的权限或安全机制。旧版关于 AST 校验、可信身份隔离、异步连接池及验证通过的说明不适用于当前实现。数据库中已有的合成测试数据未删除。
