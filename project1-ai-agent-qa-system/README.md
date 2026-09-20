# AI Agent 智能问答系统

> 面向技术文档与售后 FAQ 场景，支持多轮对话、工具调用与检索增强的问答 Agent。
> 毕业设计项目，独立完成。

## 技术栈

| 层级 | 技术选型 |
|------|---------|
| 服务层 | Python · FastAPI · Uvicorn |
| Agent 编排 | ReAct 状态机（自研） |
| 向量存储 | Milvus 2.4（HNSW + 稀疏索引） |
| 向量化 | BGE-M3（稠密 + 稀疏双向量） |
| 关键词检索 | BM25（rank-bm25 + jieba 分词） |
| 融合策略 | RRF (Reciprocal Rank Fusion, k=60) |
| 重排 | bge-reranker-v2-m3 (Cross-Encoder) |
| 语义缓存 | Redis + 向量相似度 |
| 模型路由 | Qwen2.5-7B (vLLM 本地) / DeepSeek-V2.5 (API) |
| 可观测 | OpenTelemetry → Jaeger |
| 部署 | Docker Compose |

## 架构

```
用户查询
    │
    ▼
[语义缓存] ──命中──> 直接返回
    │未命中
    ▼
[ReAct Agent 状态机]
    │
    ├─ Thought: 决定是否需要检索
    ├─ Action: 调用 RAG 工具
    │   ├─ BGE-M3 编码（稠密 + 稀疏）
    │   ├─ Milvus 稠密向量召回 (Top-20)
    │   ├─ BM25 关键词召回 (Top-20)
    │   ├─ RRF 融合两路结果
    │   ├─ Cross-Encoder 重排 Top-20 → Top-5
    │   └─ 父子块回溯（子块命中→取父块完整上下文）
    └─ Observation: 检索结果写入上下文
    │
    ▼
[模型路由]
    ├─ 常规 FAQ → Qwen2.5-7B (本地 vLLM)
    └─ 复杂推理 → DeepSeek-V2.5 (API)
    │
    ▼
[安全处理] PII 脱敏 → 返回
```

## 关键设计决策

### 1. 为什么用父子块而非固定长度切块

固定长度切分会把完整的技术描述切成两半，检索时只命中半截。

- 父块（512 字符）：提供完整上下文
- 子块（128 字符）：用于精确匹配
- 检索时先命中子块，再回溯父块

### 2. 为什么用 RRF 而非加权求和

向量召回（cosine 0~1）和 BM25 分数（0~∞）分布不可比，加权需要反复调权重。

RRF 只用排名：`score = 1/(k + rank)`，无超参负担。

| 策略 | Recall@5 |
|------|----------|
| 纯向量 | 0.71 |
| 纯 BM25 | 0.63 |
| 加权求和 | 0.78 |
| **RRF (k=60)** | **0.84** |

### 3. Agent 状态机设计

- 每步状态带版本号 + 快照，异常时回滚到上一稳定态
- 最多 8 步 + 重复工具调用检测，防止死循环
- 工具失败三级处理：重试 1 次 → 本地纠错 → 兜底话术

### 4. 安全护栏

- 用户输入与检索内容分角色拼接，降低间接 Prompt 注入风险
- 危险工具两阶段提交（先返回预览，确认后执行）
- 输出正则脱敏手机号、邮箱、身份证号

## 快速开始

### 1. 启动基础设施

```bash
docker-compose up -d milvus redis jaeger
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 文档入库

```bash
python scripts/ingest.py --input data/docs/
```

### 4. 启动服务

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 5. 测试对话

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "产品A的保修期是多久"}'
```

### 6. 运行评测

```bash
python scripts/run_eval.py --dataset data/eval/eval_dataset.json
```

## 项目结构

```
.
├── config/
│   └── settings.py           # 全局配置（所有可调参数）
├── src/
│   ├── chunking/             # 父子块切分
│   ├── embedding/            # BGE-M3 向量化
│   ├── retrieval/            # 混合检索（Milvus + BM25 + RRF）
│   ├── reranking/            # Cross-Encoder 重排
│   ├── agent/                # ReAct 状态机
│   ├── tools/                # 工具封装
│   ├── cache/                # 语义缓存
│   ├── api/                  # FastAPI 接口
│   ├── evaluation/           # 评测框架
│   └── utils/                # LLM 客户端 + 追踪
├── scripts/
│   ├── ingest.py             # 文档入库
│   └── run_eval.py           # 评测脚本
├── tests/                    # 单元测试
├── data/
│   ├── docs/                 # 文档库
│   └── eval/                 # 评测集（含对抗样本）
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 评测结果

200 条评测集（含 20 条对抗样本）：

| 指标 | 值 |
|------|----|
| Recall@5 | 0.91 |
| 忠实度 (Faithfulness) | 0.91 |
| 相关性 (Relevance) | 0.92 |
| 安全通过率 | 0.97 |
| 缓存命中率 | 0.18 |

## 走过的弯路

最初直接用固定长度切块 + 纯向量召回，实测型号、错误码这类短查询命中很差——向量擅长语义、不擅长精确匹配。改用父子块 + BM25 混合召回后才明显改善，这也是我理解混合检索价值的实际来源。

## 许可证

MIT
