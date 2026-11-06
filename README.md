# 天宫医疗智能体

项目以 FastAPI 提供对话接口，由 Supervisor 调度智慧问诊、报告解读、药物咨询、医学知识问答和运营数据五个 Worker。各 Worker 的实现范围与验证结果见[落地评估](docs/supervisor-worker-assessment.md)。

## 项目结构

- `src/` 包含 API、Supervisor、Worker、知识检索和基础设施客户端。
- `scripts/` 包含 Gradio 调试界面、数据导入和 RAG 评估入口。
- `alembic/` 管理 PostgreSQL 表结构迁移。
- `data/` 保存医学知识图谱的原始数据、指南文档和评估题集。
- `docs/` 保存项目评估和 [ChatBI 配置说明](docs/chatbi.md)。

## 本地启动

```sh
conda create -n tiangong python=3.13
conda activate tiangong
pip install -r requirements.txt
cp .env.example .env
```

在 `.env` 中填写模型密钥和服务地址。使用本地 Docker Compose 基础设施时，从项目根目录运行：

```sh
docker compose up -d
python -m alembic upgrade head
uvicorn src.main:app --port 8080 --reload
```

服务启动后访问 `http://localhost:8080/docs` 查看 API，或请求 `/health` 和 `/health/deps` 检查应用与依赖状态。

## 知识问答 RAG Gradio 测试页

项目包含一个独立的 Gradio 测试前端，直接复用 `Knowledge Agent` 的文档、图谱、SQL、多路融合和处方审核工具：

```sh
python scripts/gradio_knowledge_app.py
```

默认从 `7861` 开始寻找可用端口；若已被占用，自动尝试 `7862`、`7863` 等，实际地址以启动终端输出和自动打开的浏览器为准。可通过 `GRADIO_SERVER_PORT` 设置起始端口，通过 `GRADIO_NUM_PORTS` 设置尝试数量（默认 100）。首次提问时才初始化模型、Milvus 和 Neo4j 客户端；运行前请确认 `.env` 已配置，并启动对应的基础设施。页面支持自动工具选择、显式检索通道、多轮上下文、文档类型过滤和工具调用运行日志。前端实时显示任务阶段和进度；完整异常堆栈记录在启动终端与 `logs/` 日志文件中，页面显示简短失败提示。

若页面能打开但按钮、页签无响应，甚至浏览器内存持续增长，请确认当前 Python 环境使用 `requirements.txt` 中的 Gradio 6.16.0 及配套依赖。该版本修复了表格更新与页签选择引起的浏览器卡死（Gradio #13240）；升级后需重启 Gradio 进程并刷新页面，已运行的旧进程仍会提供旧版前端。

页面的“文档入库”页签支持上传 PDF、Word、Markdown、图片等文件，流程为：

```text
上传文件 → 本地 MinerU V1 API 解析 Markdown → SentenceSplitter 分块
→ DashScope Embedding 向量化 → upsert 到 Milvus knowledge_docs
```

默认可选择“仅使用 MinerU（失败即停止）”来验证离线 MinerU 链路，也可以选择 MinerU 失败后回退 LlamaIndex。文档类型和分类会写入 Milvus，重复上传同名文档会覆盖原分块。使用该功能前需要启动 `mineru-kit api-server`，并在 `.env` 中配置 `MINERU_API_URL`；页面的“检测 MinerU”按钮可先检查服务状态。这里的“离线”指文档由本地部署的 MinerU 解析；当前 DashScope Embedding 仍需网络和 API 密钥。

### RAG 评估与 TruLens Dashboard

两套 RAG 评估的全部评分指标统一使用 DeepSeek，由 `DEEPSEEK_MODEL`、`DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL` 配置。页面和命令行共用该配置，实验记录中的 `judge_model` 保存实际使用的 `DEEPSEEK_MODEL`。修改配置后重启评估进程。

如果兼容接口明确返回 `response_format` 不可用，评分 Provider 会自动去掉该参数重试，并在当前实例后续请求中使用文本输出，由 TruLens 解析评分；其他请求错误仍按失败处理。历史失败记录不会自动补分，需要重新运行评估。

页面的“RAG 评估”包含“文档 RAG 评估”和“多通道 RAG 评估”两个子页面。文档页集中比较 HyDE、重排序、分块策略及组合；多通道页比较文档、图谱、SQL 和文档 + 图谱融合等通道效果。评估追踪和评分仅持久化到 PostgreSQL 的 `trulens_eval` 数据库，页面仅展示固定题单、本次运行进度、简要通道均分和运行日志；逐题明细、检索链路和历史对比统一在 TruLens Dashboard 查看，不生成本地 CSV。运行前需要在 `.env` 配置的 PostgreSQL 实例中创建 `trulens_eval` 数据库。

评估页面按“固定评估题单 → 配置实验 → 执行评估”组织：左侧选择题目并确认预览，右侧设置标签、通道与指标，检索策略和预算收进折叠区；下方只保留启动/停止、本次概览与可展开的日志。选择“全部题目”会禁用抽样题数和随机种子，切换到自定义文档策略会自动展开策略控件。页面提供 Dashboard 入口，默认 `http://127.0.0.1:8501`，需单独启动。

页面先选择评估集、分类和随机抽样/全部题目，点击“生成并固定题单”查看问题及参考答案，再开始评估。切换通道、实验标签或优化策略会复用固定题单和执行顺序；修改抽样条件后须重新生成。题单保存在当前页面会话中，刷新后需要重新生成。题单确认区和运行日志显示题单编号，完整 `sample_batch_id` 保存至实验配置；比较通道时保持题单、角色、指标和检索参数一致。

也可以通过命令行运行评估，输出进度和通道均分：

```sh
python -m scripts.run_rag_evaluation --samples 1 --label baseline
```

查看完整回答、检索链路、指标评分和历史实验时，启动 Dashboard：

```sh
python -m src.agents.knowledge.evaluation.trulens_config
```

默认端口为 `8501`，可通过末尾参数修改，例如 `python -m src.agents.knowledge.evaluation.trulens_config 8502`。在 Dashboard 中选择应用 `tiangong-knowledge-rag`，不同运行和通道通过下面的 `app_version` 命名区分。Dashboard 依赖已包含在 `requirements.txt` 中。

`app_version` 统一使用 `<入口>-<系统用户>/<实验标签>/<本机时间YYYYMMDD-HHMMSS>-<8位随机编号>/<通道>/<策略>`，例如：

```text
web-violet/图谱验证/20261015-143015-a1b2c3d4/graph_rag/rw0
cli-violet/baseline/20261015-143200-e5f6a7b8/doc_rag/original-rw0-hy0-rr0-pc0-k20-n5-0123456789
```

`web` 表示从 Gradio 页面运行，`cli` 表示从命令行运行，系统用户由启动进程的操作系统账号确定。页面标签留空或命令行省略 `--label` 时使用 `manual-test`；支持中文标签，空格和路径分隔符会规范为连字符。每次开始评估都生成独立运行编号，同标签重复运行不会合并；同一次运行的多个通道共用编号。文档及融合通道保留分块、Query 改写、HyDE、重排、父块扩展和检索预算等策略后缀，其他通道用 `rw0` / `rw1` 表示改写是否开启。运行日志显示完整版本，TruLens metadata 与逐题结果的 `configuration.evaluation_run` 保存入口、用户、标签、运行编号及带时区的开始时间。已有历史版本保留原名。

评估实例按 channel 在进程内缓存：相同标签、配置和参考答案复用 TrackedKnowledge 与评分指标；每次运行会清理并替换旧 TruApp，以独立版本保存追踪。每通道最多保留当前运行的两套追踪实例（有／无参考答案），不启用后台评估线程。NL2SQL 的数据库会话仍按次打开并关闭，不进入缓存。

当前追踪事件及评分保存在 `trulens_eval.public.trulens_events`，应用版本保存在 `public.trulens_apps`；`trulens_records` 和 `trulens_feedbacks` 为空不代表没有评估数据。

每次问答的来源状态、实际使用来源、生成状态和逐项评分状态保存在同一数据库的 `public.tiangong_eval_results` 表，首次评估或刷新历史时自动建表。`attempt_id` 标识一次执行，`record_ids` 关联 TruLens 追踪；开始、回答结束和评分结束分别更新，进程中断后未完成记录仍可读取。不产生本地 CSV。

页面汇总区分“回答完成”和“评估完整”，每项均分附“有效/适用”题数。正常零召回、SQL 安全拦截和有效 0 分不算技术故障；缺少参考答案的题目逐题跳过参考答案一致性，标为 `not_applicable`。零召回的上下文相关性按规则记 0；有据性保持 TruLens 原有计算规则。来源故障、HyDE/重排等步骤故障后降级继续回答时保留评分并标记降级；所有检索来源失败时跳过评分，避免把数据库故障当成回答质量。

命令行只在适用指标完整、无技术故障降级且状态保存成功时返回退出码 0，否则返回非零退出码。非法评分在 TruLens 导出前被拒绝，缺失值不补成 0，已完成的评分保留。每次运行使用独立的实验版本；完整历史记录与跨实验对比由 TruLens Dashboard 展示。

TruLens Dashboard 展示原始追踪、逐题指标与历史对比；Gradio 仅提供本次运行的完成情况和简要均分，不查询历史排行或向页面发送逐题明细。详细状态仍保存在 `tiangong_eval_results.payload`。涉及数据库或模型异常的状态只保存稳定错误码，异常详情留在后端日志。


### 同集合的父子分块与语义分块评估

第二套知识 Agent 评估已取消混合检索，所有策略使用原 `knowledge_docs`，不创建新 collection，不删除或重写原始分块。页面“RAG 评估 → 文档 RAG 评估”默认提供“基线（无优化）”：只检索原始片段，关闭 Query 改写、HyDE、重排序和父块扩展；基线与优化模式共用相同的 `top_k` 和最终片段预算，切换策略会复用已固定题单。选择“自定义策略组合”后可配置：

- `original`：原始分块，默认选项，不需要构建版本。
- `parent_child`：子块向量召回，可选择是否扩展到父块供生成与评分；需填写已完成构建的版本。
- `semantic`：按语义切分后的片段检索；需填写已完成构建的版本。

父子与语义是互斥的分块策略，不是两个可叠加的查询开关。Query 改写、HyDE、Rerank 仍可独立控制，召回数量和最终片段数量仍可配置。“多通道 RAG 评估”页的文档通道固定使用 `original`（固定分块）、开启 HyDE 和重排序；融合中的文档检索复用相同的 `doc_rag.py` 实现和选项。该页可配置文档类型与片段预算，文档类型对融合中的文档检索也生效。页面默认值仅在多通道评估入口应用，底层检索函数、`DocRetrievalOptions` 及原有 CLI 的默认值和参数保持兼容。

父子与语义策略使用已有的分块版本。构建清单保存在 PostgreSQL `trulens_eval.public.tiangong_doc_chunk_builds`，只有状态为 `ready` 且与当前集合、分块策略及记录数量一致的版本可以评估。

后续上传、药品说明书和公共数据集入库直接携带原始分块元数据：`chunk_strategy=original`、`index_version=original_v1`、`source_id=自身主键`；`parent_id` / `parent_text` 使用空字符串表示“不适用”。原始策略评估不需要填写版本。

#### 选择策略及版本评估

```sh
python -m scripts.run_rag_evaluation --channels doc_rag --samples 10 --seed 42 --label parent-test --chunking parent_child --index-version parent_v1
python -m scripts.run_rag_evaluation --channels doc_rag --samples 10 --seed 42 --label semantic-test --chunking semantic --index-version semantic_v1
```

父子模式加 `--no-parent` 可只使用召回子块，用于隔离“父块扩展”的影响。默认 `--chunking original`；不再提供 `--hybrid`、`--collection` 参数。

在 Dashboard 的 `tiangong-knowledge-rag` 应用中比较各实验版本。策略、构建版本、HyDE/重排、Query 改写和 `top_k` / `context_top_k`（传给检索的 `rerank_top_k`）会显示在运行日志中，完整配置写入 TruLens metadata 和 PostgreSQL 逐题结果的 `configuration`。启用 HyDE 必须提供生成模型；索引缺少策略字段、版本不匹配、未完整构建或数量不完整时，页面会明确提示并拒绝评估。对比时固定题目、随机种子、模型及语料范围；原始分块默认搜索全部原始语料，因此不要把“小样本新分块”与“全量原始分块”的分数差直接归因于分块策略。使用不同实验标签可分开记录同一配置的重复运行。

### 解析产物、图片与分块预览

入库继续复用 `ingest_file`：MinerU（失败可回退 LlamaIndex）→ 保存原始解析 → 提取内嵌图片并清理正文 → Markdown 结构分块（失败回退 SentenceSplitter）→ 每批最多 20 条向量化 → 原 Milvus upsert。Milvus 字段和同名覆盖规则保持现有行为。

解析产物默认保存在 `data/parsed/<doc_id>/`，其中 `doc_id` 仍为文档名 MD5 的前 16 位。目录含 `raw.md`、`document.md`、`retrieval.md`、`documents.json`（保留回退解析的页码）、`assets.json`、`images/`、`chunks.jsonl` 和 `manifest.json`。MinIO 使用已配置的 `MINIO_BUCKET`，对象前缀为 `knowledge/artifacts/<doc_id>/`。`PARSED_ARTIFACT_DIR` 可修改本地目录，相对路径基于项目根目录。

Markdown、HTML 和引用式 Base64 图片提取后单独保存；检索正文保留图注，不包含图片编码。章节前缀加入分块正文，表格分块重复表头。原始解析结果保留，便于排查和下载。图片尚不做 OCR 或视觉模型理解。

每次上传都重新解析、分块和向量化，不读取已有的本地或 MinIO 解析缓存；Embedding 失败后重传也会重新解析。解析产物仍保留供预览和下载，预览时校验文件内容是否与产物一致。MinIO 故障时保留本地产物并继续；本地保存失败则直接将内存中的产物字节上传 MinIO，无需依赖临时目录，两边均无法保存时停止向量化。

Gradio“文档入库”页签移除“复用同文件解析缓存”和“强制重新解析”开关，保留“解析产物与分块预览”：原始/检索 Markdown、图片、分块表、产物清单和下载。预览显示本地副本，不渲染原始 HTML 或 SVG；预览截取前 30,000 字符，下载以 ZIP 提供正文和清单全文。图片预览验证并重新编码为 PNG；不开放整个产物目录。完整异常仍只在后端输出。

## 主 Agent / 子 Agent 模型配置

模型配置统一由 `src/core/config.py` 读取项目根目录的 `.env`，不在 Agent 文件中硬编码模型名称。
部署时同名进程环境变量优先于 `.env`；修改配置后需重启服务（Settings 和 Agent 都有缓存）。

| 调用范围 | 模型配置 | 密钥配置 | 请求地址 |
| --- | --- | --- | --- |
| 主 Agent（Supervisor、会话摘要） | `CHAT_MODEL` | `DASHSCOPE_API_KEY` | `BASE_URL_CHAT` |
| 子 Agent（问诊图、挂号、药物、报告、知识、运营） | `DEEPSEEK_MODEL` | `DEEPSEEK_API_KEY` | `DEEPSEEK_BASE_URL` |
| 长期记忆、症状向量检索 | `EMBEDDING_MODEL` | `DASHSCOPE_API_KEY`，或 `EMBEDDING_API_KEY` | DashScope 向量接口，或 `EMBEDDING_BASE_URL` |

新主 Agent 使用 `get_supervisor_llm()`，新子 Agent 使用 `get_worker_llm()`。
显式设置 `DEEPSEEK_BASE_URL`，并填写该服务支持的 `DEEPSEEK_MODEL` 和对应密钥。主、子 Agent 使用同一兼容网关时，将该地址同时填入 `BASE_URL_CHAT` 和 `DEEPSEEK_BASE_URL`。
`VL_MODEL` 保留供未来视觉能力接入，当前报告解读仍走子 Agent 文本模型。

修改向量模型时，需要确认已有 Milvus 索引的维度和向量数据与新模型一致，必要时重建索引。


Mineru部署清单（Mac / conda + pip）
# 1. 环境（已存在可跳过）
conda create -n mineru python=3.12 -y
conda activate mineru

# 2. 安装 4.x（Apple Silicon 会自动带上 Torch，不需要 [full]；3.x 的 [core]/[pipeline] 等 extras 已废弃）
pip install -U "mineru>=4.0,<5" -i https://mirrors.aliyun.com/pypi/simple
mineru version --json          # 确认是 4.x

# 3. 下载并校验 standard 档位模型（国内用 modelscope）
export MINERU_MODEL_SOURCE=modelscope
mineru-kit models download --tier standard --source modelscope
mineru-kit models verify --tier standard

# 4. 命令行先试一份 PDF
mineru-kit parse 某指南.pdf -o 某指南.md --tier standard

# 5. 启动 API 服务（mineru-api 是同一个入口）
mineru-kit api-server --host 127.0.0.1 --port 8000 --tier standard
# 5.1. 直接解析 PDF
mineru-kit parse input.pdf -o ./output.md
# 5.2. 启动网页界面
mineru-kit webui --server-name 127.0.0.1 --server-port 7860
# 5.3. 启动文档解析 API
mineru-kit api-server --host 127.0.0.1 --port 8000 --tier standard
# 5.4. 启动 VLM 推理服务
mineru-kit vlm-server --engine auto --port 30000


# 6. 验证
curl http://127.0.0.1:8000/v1/health    # 接口文档在 http://127.0.0.1:8000/docs

## ChatBI 自然语言分析

`src/agents/nl2sql` 保存当前 ChatBI 实现，包括自然语言查询、多轮上下文、SQL 校验、图表推荐以及 Gradio 页面；未注册到 `src/main.py`。上一版实现的测试、验证报告、测试数据脚本和不兼容的独立启动脚本已清理。

本次仅整理并提交代码，未运行测试或验证启动。当前入口及已知接入问题见 [ChatBI 说明](docs/chatbi.md)。
