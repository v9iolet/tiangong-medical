# 天宫医疗 Agent 落地评估

评估日期为 2026 年 11 月 5 日，基于仓库版本 `dc17876` 和本地测试。

## 结论

**后端已有 Supervisor 和 5 个 Worker 的调度入口，但尚未完成可验收的「1 个 Supervisor + 3 个 Worker」闭环。** 知识问答、智慧问诊和药物咨询都有实际业务代码，适合作为首期候选。三者的端到端验收仍需补齐。报告解读目前接收文本并由模型直接生成回答，运营数据入口缺少 SQL 工具必需的数据库会话。

目前没有独立业务前端。现有 Gradio 主要用于验证知识检索、文档导入、处方审核和 RAG 评估。它直接构建 Knowledge Agent 或调用工具，页面能回答知识问题也无法证明 Supervisor 路由正确。[知识 Gradio](../scripts/gradio_knowledge_app.py#L325)

## 项目入口与调用关系

```text
知识 Gradio → Knowledge Agent / 指定检索工具

FastAPI POST /api/v1/chat 或 /api/v1/chat/stream
  ├─ Redis 存在活跃问诊 → InquiryGraph 续轮 → 回复或问诊移交
  └─ 无活跃问诊 → Supervisor → 按意图调用 Worker → 汇总回复
                   ├─ Redis Stack checkpoint 保存会话
                   └─ Milvus 保存和检索长期记忆
```

- **后端**使用 FastAPI，Supervisor 用 LangChain `create_agent` 把 Worker 封装成工具。问诊使用 LangGraph 状态机，后续轮次由 API 根据 Redis 状态直达问诊图。普通聊天支持同步返回完整文本和 SSE 两种接口，活跃问诊在 SSE 中整段返回。[应用入口](../src/main.py#L87) · [聊天路由](../src/api/routers/chat.py) · [Supervisor](../src/agents/supervisor_agent.py#L57)
- **数据层**由 PostgreSQL 承载业务数据，Redis Stack 管理会话和问诊状态，Milvus 支撑向量检索和长期记忆，Neo4j 提供疾病、症状等关系，MinIO 与文档处理链路配套。[基础设施编排](../docker-compose.yml) · [问诊工作流](../src/agents/inquiry/graph.py)
- **其他入口**包括知识管理路由，以及独立的 ChatBI 路由和 Gradio 源码。当前主应用只挂载聊天业务路由。ChatBI 仍有指向 `src.nl2sql` 的旧导入路径，运营 Worker 走的是知识检索中的 SQL 工具，两条链路应分开验收。[知识管理路由](../src/api/routers/knowledge.py) · [ChatBI 说明](chatbi.md)

## Worker 现状

| Supervisor 注册入口 | 已有能力 | 影响验收的缺口 |
| --- | --- | --- |
| `call_inquiry_agent` | 首轮启动问诊图，后续可恢复状态并追问，生成分诊建议和移交数据 | 后续轮次跳过急症检查。文件中有两个同名 `handle_handoff`，实际调用后定义的模拟实现，会返回预约成功和模拟预约号。[问诊图](../src/agents/inquiry/graph.py#L96) · [移交实现](../src/agents/workers/inquiry_agent.py#L100) |
| `call_knowledge_agent` | 文档 RAG、GraphRAG、融合检索和处方审核都有工具实现 | Supervisor 经全局实例使用默认用户和角色，调用只传文本。SQL 工具虽已注册，默认实例没有数据库会话。[知识 Worker](../src/agents/workers/knowledge_agent.py#L61) |
| `call_drug_agent` | 复用文档、图谱、融合检索及处方审核 | 和知识 Worker 在药品资料、用药安全、处方审核上有重叠。需明确路由边界，并把结构化审核结果、证据及资料缺失状态保留到最终回复。[药物 Worker](../src/agents/workers/drug_agent.py) · [Worker 封装](../src/agents/tools/worker_tools.py#L81) |
| `call_report_agent` | 基于模型解释传入的文本 | Chat API 没有报告附件字段，Worker 没有文件解析、视觉或历史报告对比工具。当前可作为文本演示。[报告 Worker](../src/agents/workers/report_agent.py#L23) · [请求模型](../src/api/routers/chat.py#L24) |
| `call_operation_agent` | 注册运营 SQL 工具 | 默认实例没有数据库会话，工具直接返回连接不可用。运营授权、数据范围与聚合约束需要在应用和数据库层落实。[运营 Worker](../src/agents/workers/operation_agent.py#L40) · [SQL 工具](../src/agents/knowledge/tools.py#L83) |

## 优化顺序与验收标准

1. **P0 · 明确首期三个 Worker。** 建议选择「问诊分诊、医学知识、药物咨询」，以现有代码量和业务可验证性为依据。症状和就诊科室进问诊，通用指南和文献进知识，药品使用、相互作用和处方审核进药物。按部署配置限制首期暴露的 Worker，并让高优先级分支由应用层控制。路由回归应覆盖含糊输入、跨领域请求和活跃问诊中切换话题。若首期指定运营或报告，需先补齐该 Worker 在上表中的接入条件。[注册表](../src/agents/tools/worker_tools.py#L123) · [路由提示词](../src/agents/supervisor_agent.py#L13)
2. **P0 · 修复身份和医疗流程边界。** API 应从认证主体派生用户与角色，并校验患者档案访问权限。Supervisor 和 Worker 使用一致的可信上下文，统一账号记忆与患者档案记忆的命名空间及访问策略。每轮都经过急症分流，模型或解析失败时提供明确兜底。问诊移交先标记为分诊建议或模拟状态，真实预约以预约系统的确认结果为准。聊天失败向客户端返回稳定错误码，堆栈留在服务端，移除日志中的原始病史文本。验收覆盖越权访问、问诊中出现紧急情况、移交失败和日志脱敏。[聊天路由](../src/api/routers/chat.py) · [记忆工具](../src/agents/tools/store_tools.py#L12) · [问诊写回](../src/agents/inquiry/graph.py#L423)
3. **P1 · 统一对话执行与 Worker 返回契约。** 让普通接口和 SSE 共用同一轮执行逻辑。原始用户输入、授权上下文和检索证据分字段传递，Worker 返回结构化的回答、来源、业务状态及错误码。将问诊活跃标记与状态合并为单键或原子更新，并为同一会话的并发请求、问诊记录和移交建立幂等机制。当前续轮在移交成功前清除 Redis 状态，移交异常后缺少可重试状态。验收要求普通接口与 SSE 的最终回复一致，工具中间输出不会混入用户回答，同一轮重试不会重复写记录或移交。[聊天执行](../src/api/routers/chat.py#L40) · [流式过滤](../src/agents/streaming.py)
4. **P1 · 用统一 API 完成验收，再接业务前端。** 先修复旧测试导入，统一 Worker 模型配置文档和实际行为。增加通过 FastAPI 真实入口的回归，分别校验三个 Worker 的命中、问诊跨轮、证据与来源保留、依赖失败及用户隔离。测试覆盖应包含可复现的替身用例和独立测试环境中的真实模型及数据服务。可先用现有 Gradio 新增一个 API 联调入口，验证前端消费的会话、SSE 和错误事件契约，再开发正式前端。
5. **P2 · 扩展报告和运营能力。** 报告接入带访问控制的上传及解析，保留原文、单位与参考区间。运营接入请求级只读数据库会话，使用受限数据视图和服务端授权校验，再验证 SQL 数据范围、超时和空结果。若选 ChatBI 作为运营实现，先修正模块导入并明确与现有知识 SQL 工具的边界。[报告 Worker](../src/agents/workers/report_agent.py) · [知识 SQL 校验](../src/agents/knowledge/retrieval/nl2sql.py#L39) · [ChatBI 说明](chatbi.md)

## 本次验证

在 Conda 的 `tiangong` 环境运行仓库现有本地测试。问诊身份与状态、流式过滤、知识 Gradio 三个测试文件共 **48 通过**。这些结果覆盖部分组件行为。

- 使用 FastAPI `TestClient` 读取 OpenAPI 返回 HTTP 200，确认两个聊天入口和两个健康检查入口已注册。检查 Worker 工具列表得到 5 个入口。导入 ChatBI Gradio 模块时复现 `ModuleNotFoundError: src.nl2sql`。
- `test/test_supervisor_agent.py` 单独收集失败，原因是仍从 `agents.supervisor_agent` 导入，当前包路径为 `src.agents`。
- 排除该收集错误后，整套本地测试为 **489 通过、5 失败、5 跳过**。失败均在 `test/test_agent_model_config.py`，包括 Worker 地址留空时未回退至 `BASE_URL_CHAT`（与 README 所述不一致）、三个 Worker 的温度参数调用与测试期望不一致，以及问诊依赖测试未替换新增的 `MilvusStore`。
- 本次验证范围是源码、路由注册和本地测试。真实模型、数据服务、三 Worker 端到端对话和预约系统仍待联调验收。
