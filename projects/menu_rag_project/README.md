# Menu RAG Project — 菜谱问答系统

一个基于 **LangChain + FAISS + 通义千问（Qwen）** 的菜谱检索增强生成（RAG）问答系统。内置 323 份中文菜谱（Markdown 格式），支持按菜品分类、难度进行元数据过滤，采用「混合检索 + 重排序 + 父子文档回溯」的经典 RAG 优化方案。

## 功能特性

- **混合检索**：向量检索（FAISS）+ BM25 关键词检索双路召回，结果融合
- **重排序（Rerank）**：使用 CrossEncoder（`BAAI/bge-reranker-v2-m3`）对初筛结果精排
- **父子文档（Parent-Child Chunking）**：按 Markdown 标题切分为小块用于检索，命中后回溯到完整食谱文档再交给 LLM，兼顾检索精度与上下文完整性
- **元数据过滤**：自动识别问题中的分类（荤菜/素菜/汤品等 9 类）与难度（简单~非常困难），检索前先过滤
- **查询路由与改写**：根据问题类型（列表型/详细型/通用型）分发到不同生成策略，非列表型问题先做查询改写
- **流式输出**：支持逐 token 打印回答

## 项目结构

```
menu_rag_project/
├── main.py                  # 入口：RecipeRAGSystem 编排类 + 交互式问答循环
├── config.py                # RAGConfig 配置（数据路径、模型、top_k 等）
├── pyproject.toml            # uv 项目与依赖声明
├── uv.lock                   # 依赖锁定文件（uv sync 还原）
├── requirements.txt          # 依赖清单（pyproject 同源，兼容 pip）
├── .env                     # API 密钥与模型配置（不入库）
├── rag_modules/
│   ├── data_preparation.py      # 数据准备：加载 323 份菜谱 md，父子分块，提取元数据
│   ├── index_construction.py    # 索引构建：bge-small-zh-v1.5 向量化 + FAISS 索引存取
│   ├── retrieval_optimization.py# 检索优化：向量+BM25 混合检索、CrossEncoder 重排、元数据过滤
│   └── generation_integration.py# 生成集成：Qwen LLM、查询路由/改写、三种回答模板、流式输出
├── data/                    # 菜谱知识库（323 个 .md，按 9 个分类目录组织）
└── vector_index/            # 预构建的 FAISS 索引（index.faiss / index.pkl）
```

## 技术栈

| 环节 | 技术选型 |
|------|---------|
| Embedding | `BAAI/bge-small-zh-v1.5`（HuggingFace，CPU，本地推理） |
| 向量库 | FAISS（langchain-community） |
| 关键词检索 | BM25（rank-bm25） |
| 重排序 | CrossEncoder `BAAI/bge-reranker-v2-m3` |
| LLM | 通义千问 `qwen-max`（DashScope OpenAI 兼容接口，langchain-openai） |
| 文本切分 | MarkdownHeaderTextSplitter（langchain-text-splitters） |

## 环境准备

项目使用 [uv](https://docs.astral.sh/uv/) 管理依赖（`pyproject.toml` + `uv.lock`）。

### 1. 创建虚拟环境并安装依赖（Python 3.12+）

```bash
cd projects/menu_rag_project
uv sync          # 按 uv.lock 精确还原依赖到 .venv
```

> 首次运行会从 HuggingFace 下载 embedding 与 reranker 模型（各几百 MB），建议提前配置代理或 `HF_ENDPOINT` 镜像。

### 2. 配置 API 密钥

编辑项目根目录的 `.env`：

```ini
OPENAI_API_KEY=sk-xxxxxxxx        # 阿里云百炼平台的 API Key
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-max                # 可改为 qwen-plus
```

API Key 申请：[阿里云百炼平台](https://bailian.console.aliyun.com/) → API-KEY 管理。

## 运行

```bash
cd projects/menu_rag_project
source .venv/bin/activate
python main.py
```

> 注意：`config.py` 中数据与索引均为相对路径（`./data`、`./vector_index`），**必须在项目根目录下启动**。

启动后系统会加载已有 FAISS 索引（若不存在则重新构建并保存），随后进入交互式问答：

```
🤖 请输入你的问题（或 'exit' 退出）: 简单的早餐有什么推荐？
是否使用流式输出? (y/n, 默认y): y
```

### 示例问题

- 列表型：`有哪些简单的素菜？`（走元数据过滤 + 列表生成）
- 详细型：`红烧肉怎么做？详细说说步骤`（走查询改写 + 分步生成）
- 通用型：`炖汤有什么技巧？`

## 处理流程

```
用户问题
  │
  ├─ 查询路由（list / detail / general）
  ├─ 查询改写（非列表型）
  ├─ 元数据过滤识别（分类 / 难度命中则先过滤）
  │
  ├─ 混合检索：FAISS 向量召回 ∪ BM25 关键词召回（各 top 5）
  ├─ CrossEncoder 重排 → top_k(3) 子块
  ├─ 父子文档回溯：子块 → 完整食谱文档
  │
  └─ LLM 生成（列表 / 分步 / 通用 模板，支持流式）
```

## 常见问题

- **索引重建**：删除 `vector_index/` 目录后重新运行即可（需重新 embed 全部文档）。
- **模型下载慢**：设置 `export HF_ENDPOINT=https://hf-mirror.com`，或配置代理。
- **代理下报 502 / 模型加载失败**：模型已缓存在 `~/.cache/huggingface` 时，加 `HF_HUB_OFFLINE=1` 强制离线加载，跳过联网版本校验：
  ```bash
  HF_HUB_OFFLINE=1 python main.py
  ```
- **API 报错**：确认 `.env` 中的 Key 有效且 `OPENAI_API_BASE` 未被改动。
