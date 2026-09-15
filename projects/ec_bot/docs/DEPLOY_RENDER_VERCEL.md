# 部署手册：Render（后端）+ Vercel（前端）

本文档描述如何把本项目以**零成本**部署到线上，架构为前后端分离：

```
浏览器
  │
  ├─► 前端静态站点（Vercel，免费，不休眠）
  │     构建时把 VITE_API_BASE_URL 打进产物
  │
  └─► 后端 API（Render 免费 Docker 实例，512MB）
        │
        ├─► MySQL（Aiven 免费版，1GB）—— 元数据 + 数仓 + 字段取值全文索引
        ├─► Qdrant Cloud（免费 1GB 集群）—— 字段与指标向量
        ├─► Embedding API（OpenAI 兼容）—— 向量化
        └─► LLM API（OpenAI 兼容）—— 关键词扩展、SQL 生成与修正
```

## 相对本地开发版做了哪些改造

本地 `docker/docker-compose.yaml` 会起 MySQL、Elasticsearch（带 IK 分词）、Kibana、Qdrant、TEI 推理服务。其中 ES 和 1GB+ 的本地 Embedding 模型无法放进免费托管环境，因此做了这些替换：

| 组件 | 本地开发 | 线上免费版 | 切换方式 |
| --- | --- | --- | --- |
| 字段取值召回 | Elasticsearch + IK 分词插件 | MySQL ngram 全文索引 | `VALUE_STORE_PROVIDER=mysql` |
| 向量化 | 本地 TEI + bge-large-zh-v1.5 | OpenAI 兼容 Embedding 接口 | `EMBEDDING_PROVIDER=openai` |
| 向量库 | 本地 Qdrant 容器 | Qdrant Cloud 免费集群 | `QDRANT_PROVIDER=cloud` |
| MySQL | 本地容器 | Aiven 免费实例（强制 TLS） | `META_DB_SSL=true` |
| 跨域 | 不需要（Vite dev proxy 同源） | 需要（Vercel ↔ Render 不同源） | `CORS_ALLOW_ORIGINS` |

所有外部依赖的地址与凭据都已改成环境变量注入（见 `conf/app_config.yaml` 与 `.env.example`）。**没有配置环境变量时，默认值仍然指向本地 docker-compose**，所以本地开发流程完全不受影响。

新增的代码分层：`app/repositories/value/` 定义了取值检索的统一协议（`ValueRepository`），ES 与 MySQL 两套实现通过工厂按配置切换，上层（依赖注入、召回节点、知识库构建）不再感知具体存储。

另外，`conf/app_config.yaml` 里写成 `${oc.env:XXX}`（不带默认值）的变量都是**必填项**，目前只有三个：`MODEL_NAME`、`LLM_API_KEY`、`LLM_API_BASE`。缺失时应用会在启动前直接打印缺失清单并退出，不会留下一段难读的 omegaconf 堆栈。

## 一、准备四个免费账号

### 1. Aiven MySQL（数据库）

- 免费额度：**1GB 存储 / 1GB 内存 / 1 CPU**，单节点，无到期时间，不需要绑卡
- 每个账号每种服务类型可开 1 个免费实例
- ⚠️ **闲置一段时间后服务会自动关机**（会提前邮件通知），需要去控制台手动重启

创建步骤：控制台 → Create Service → MySQL → 选免费计划 → 选离你近的区域（如 Singapore）→ 创建。
创建完成后在 Overview 页记下 **Host / Port / User / Password**，数据库 `defaultdb` 可直接使用，或自行建 `meta` 与 `dw` 两个库。

> 连接信息里的 `Port` 通常是 `2xxxx` 这类非 3306 端口，别填错。托管实例**强制 TLS**，所以后面要开 `META_DB_SSL=true`。

### 2. Qdrant Cloud（向量库）

- 免费额度：**1GB RAM / 4GB 磁盘 / 0.5 vCPU** 单节点集群，无需绑卡
- ⚠️ **连续 7 天无请求会自动暂停**，暂停后集群不响应，需在控制台手动唤醒（约 30–60 秒）
- ⚠️ 暂停超过 **4 周**未唤醒会被**删除**。收藏夹里留个提醒，或者偶尔戳一下控制台

本项目的向量数据非常小（只有字段与指标元数据，几十条），1GB 完全够用。

创建后记下 **Cluster URL**（形如 `xxxxx.cloud.qdrant.io`，不要带 `https://`）和 **API Key**。

### 3. Embedding API（必须 1024 维）

默认维度是 `EMBEDDING_SIZE=1024`，因此要选一个**输出 1024 维**的模型，例如：

- 国内：硅基流动（SiliconFlow）的 `BAAI/bge-m3`、阿里云百炼的 `text-embedding-v3`
- 国外：任意 OpenAI 兼容网关

记下 **Base URL**（如 `https://api.siliconflow.cn/v1`）和 **API Key**。

> ⚠️ **维度必须与 `EMBEDDING_SIZE` 一致**。Qdrant 集合的维度在创建时固定，改维度必须删掉集合重建索引（见第五节）。
>
> ⚠️ 换成别的 embedding 模型后，**向量空间变了，必须重新构建索引**，否则召回结果全是错的。查询向量和入库向量必须来自同一个模型。

### 4. LLM API

任意 OpenAI 兼容接口即可，记下 **模型名 / Base URL / API Key**。这个项目的 Agent 图会大量调用 LLM（关键词扩展、SQL 生成与修正），是主要耗时来源。

## 二、导入数据库结构

`docker/mysql/` 下是给本地容器用的初始化脚本，里面带有 `CREATE DATABASE` / `GRANT` / `USE` 这类需要超级权限的语句，托管实例上执行会报错。

因此已生成一份去掉这些语句的版本：

- `docker/mysql/remote/meta_schema.sql`
- `docker/mysql/remote/dw_schema.sql`

导入方式：

```bash
# 先在 Aiven 控制台（或用 SQL 客户端）建好 meta 与 dw 两个库
# 然后用 mysql 客户端导入；注意端口是 Aiven 给的非 3306 端口，必须加 --ssl-mode=REQUIRED
mysql -h <META_DB_HOST> -P <META_DB_PORT> -u <USER> -p --ssl-mode=REQUIRED meta < docker/mysql/remote/meta_schema.sql
mysql -h <DW_DB_HOST> -P <DW_DB_PORT> -u <USER> -p --ssl-mode=REQUIRED dw < docker/mysql/remote/dw_schema.sql
```

`dw_schema.sql` 自带建表与样例数据（订单事实表 + 四个维度表），约 400 行，导入后可直接问数。

> 如果 `meta` 与 `dw` 想开在同一个 Aiven 实例上，把 `DW_DB_*` 填成和 `META_DB_*` 一样的值、只改库名即可。这样能省下一个免费实例名额。

## 三、在本地构建云端索引

索引构建脚本会连接你刚准备好的**云端** Qdrant 与 MySQL，把字段/指标向量和字段取值写进去。这一步在**本地跑**，不要放到 Render 上（免费实例 512MB 内存跑不动，也会拖慢冷启动）。

```bash
cd projects/ec_bot

# 1. 复制环境变量样例，填入上面收集到的云服务凭据
cp .env.example .env
#    至少需要填：
#    MODEL_NAME / LLM_API_KEY / LLM_API_BASE
#    META_DB_HOST / META_DB_PORT / META_DB_USER / META_DB_PASSWORD / META_DB_SSL=true
#    DW_DB_HOST   / DW_DB_PORT   / DW_DB_USER   / DW_DB_PASSWORD   / DW_DB_SSL=true
#    QDRANT_PROVIDER=cloud / QDRANT_HOST / QDRANT_API_KEY / QDRANT_USE_HTTPS=true
#    EMBEDDING_PROVIDER=openai / EMBEDDING_MODEL / EMBEDDING_API_KEY / EMBEDDING_API_BASE
#    EMBEDDING_SIZE 必须等于所选 embedding 模型的输出维度（默认 1024）

# 2. 执行构建：写入 Qdrant 集合 + MySQL 取值索引表
uv run python -m app.scripts.build_meta_knowledge -c conf/meta_config.yaml
```

脚本会依次完成：

1. 建 `column_info_collection`、`metric_info_collection` 两个 Qdrant 集合（维度取 `EMBEDDING_SIZE`）
2. 把 `conf/meta_config.yaml` 里的字段与指标做向量化并写入
3. 建 `value_index` 表并用 `WITH PARSER ngram` 建全文索引，把 `sync: true` 的字段真实取值写进去

跑完后建议去 Qdrant 控制台确认两个集合有点数，MySQL 里 `select count(*) from value_index` 有数据。

> 如果目标 MySQL 建不出 ngram 全文索引（少数托管版本不支持），代码会自动退化为 `LIKE` 匹配，取值召回依然可用，只是候选质量略降。日志里会看到一条 warning。

## 四、部署后端到 Render

1. 把仓库推到 GitHub（`render.yaml` 已在仓库根目录 `render.yaml`，即 monorepo 的 `agent/` 下）
2. Render 控制台 → **New → Blueprint** → 选择该仓库 → Render 会读取 `render.yaml`
3. `render.yaml` 里已声明后端服务（`rootDir: projects/ec_bot`），需要**手动填写**的变量（`sync: false` 的项）：
   - `MODEL_NAME` / `LLM_API_KEY` / `LLM_API_BASE`
   - `META_DB_HOST` / `META_DB_PORT` / `META_DB_USER` / `META_DB_PASSWORD`
   - `DW_DB_HOST` / `DW_DB_PORT` / `DW_DB_USER` / `DW_DB_PASSWORD`
   - `QDRANT_HOST` / `QDRANT_API_KEY`
   - `EMBEDDING_MODEL` / `EMBEDDING_API_KEY` / `EMBEDDING_API_BASE`
   - `CORS_ALLOW_ORIGINS`（**先留空，等前端部署拿到域名后再回填**，见第六节）
4. 等待构建完成，记下服务地址，形如 `https://ec-bot-api.onrender.com`
5. 访问 `https://ec-bot-api.onrender.com/health`，返回 `{"status":"ok"}` 即表示存活

已经预设好的变量无需改动：`META_DB_SSL=true`、`DW_DB_SSL=true`、`QDRANT_PROVIDER=cloud`、`QDRANT_USE_HTTPS=true`、`EMBEDDING_PROVIDER=openai`、`VALUE_STORE_PROVIDER=mysql`、连接池 `*_DB_POOL_SIZE=3`、`LOG_FILE_ENABLE=false`。

> 连接池设成 3 是因为 Aiven 免费实例的连接数很紧，meta 与 dw 各 3 条共 6 条比较安全。

## 五、部署前端到 Vercel

前端产物是纯静态文件，Vercel 免费版**不休眠、不限流量额度**（个人项目规模），比放在 Render 更合适。

1. Vercel 控制台 → **Add New → Project** → 导入同一个仓库
2. **Root Directory 必须设为 `projects/ec_bot/frontend`**（monorepo 关键一步）
3. Framework Preset 会自动识别为 Vite；构建配置已由 `frontend/vercel.json` 提供，无需手填：
   ```json
   {
     "framework": "vite",
     "installCommand": "pnpm install --frozen-lockfile",
     "buildCommand": "pnpm build",
     "outputDirectory": "dist"
   }
   ```
4. 在 **Environment Variables** 里添加：
   - `VITE_API_BASE_URL` = 第四步拿到的 Render 后端地址，如 `https://ec-bot-api.onrender.com`（**不要以 `/` 结尾**）
5. 部署，拿到前端域名，形如 `https://ec-bot.vercel.app`

> 前端没有使用任何客户端路由（只有一个聊天页），所以这里**刻意没有配 catch-all 重写**——`/(.*) → /index.html` 会把 `/assets/*.js` 也重写成 HTML，反而把静态资源打坏。

> `VITE_API_BASE_URL` 是在**构建时**被写进 JS 产物的，修改后必须重新部署（Redeploy）才生效。

## 六、回填 CORS 并验证

这是前后端分离架构最容易漏掉的一步：Vercel 域名与 Render 域名不同源，浏览器会直接拦掉 `/api/query` 请求，前端表现为"请求失败"但后端日志里**什么都没有**（请求根本没发出去）。

回到 Render → 后端服务 → Environment，设置：

| 变量 | 值 |
| --- | --- |
| `CORS_ALLOW_ORIGINS` | `https://ec-bot.vercel.app`（你的前端正式域名；多个用英文逗号分隔） |
| `CORS_ALLOW_ORIGIN_REGEX` | 可选。Vercel 每次推送都会生成新的预览域名，用 `https://ec-bot-.*\.vercel\.app` 一次覆盖 |

保存后 Render 会自动重新部署。然后就完整验证一遍：

1. `curl https://ec-bot-api.onrender.com/health` → `{"status":"ok"}`
2. 打开前端页面，问一句「统计 2025 年第一季度各大区的 GMV，并按 GMV 从高到低排序」
3. 观察左边的步骤轨（关键词抽取 → 三路召回 → 合并 → 过滤 → 生成 SQL → 校验 → 执行）

如果第 3 步失败，按下面顺序排查：

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 前端报跨域错误，后端无日志 | `CORS_ALLOW_ORIGINS` 没填或填错（多了 `/`、少了协议头） | 必须写全 `https://` 且不带结尾斜杠 |
| 首次访问卡 50 秒以上 | Render 免费实例休眠后被唤醒 | 正常现象，见第七节 |
| 后端日志报连不上 MySQL | Aiven 实例闲置自动关机了 | 去 Aiven 控制台重启服务 |
| 后端日志报 Qdrant 连接失败 | Qdrant 免费集群 7 天无活动被暂停 | 去 Qdrant 控制台唤醒 |
| 向量召回结果离谱 | 换了 embedding 模型但没重建索引 | 删掉两个 Qdrant 集合，重跑第三节的构建脚本 |
| 报维度不匹配 | `EMBEDDING_SIZE` 与模型输出维度不一致 | 改成模型真实维度并重建集合 |
| 容器启动即退出，日志提示「配置加载失败：以下必填环境变量未设置」 | 变量没填全 | 照着日志里列出的名字，逐个补到 Render 的 Environment 里 |
| 容器启动即退出，日志里是 `omegaconf.errors.InterpolationResolutionError` | 变量名拼错了（缺失的变量会被上面那条检查拦下，能走到这里说明是写错了名字） | 对照 `conf/app_config.yaml` 核对拼写 |

## 七、免费额度的现实约束

部署前请确认这些是你能接受的：

- **Render 免费实例会在 15 分钟无请求后休眠**，下一个请求需要等约 50 秒冷启动。面试演示场景建议提前打开页面预热一下。
- **Render 免费实例只有 512MB 内存 / 0.1 CPU**，这个项目的 Agent 图要多次调用 LLM，单次问数响应会比较慢（十几秒到几十秒量级），属于正常表现。
- **Aiven 免费 MySQL 闲置会自动关机**，需要手动重启；只有 1GB 存储，样例数据量很小所以够用。
- **Qdrant 免费集群 7 天无活动暂停、4 周未唤醒删除**，向量数据要做好"随时能重建"的准备——好消息是重建只需跑一次第三节的脚本，几分钟即可完成。
- Render 免费版目前已不再提供免费 Postgres，所以本方案完全不依赖它。

如果哪天需要真正的 7×24 在线，可以把后端换成一台轻量云主机（2核4G，约 ¥30–60/月），用 `docker/docker-compose.yaml` 原样拉起全栈，代码零改动——所有地址都是环境变量注入的，这也是这次改造顺带带来的好处。

## 八、本地开发不受影响

回填的变量都有本地默认值，所以不配 `.env` 也能按老方式开发：

```bash
cd docker && docker compose up -d     # 起 MySQL / Qdrant / ES / TEI
cd .. && uv run uvicorn main:app --reload
cd frontend && pnpm dev               # Vite dev proxy 把 /api 转发到 127.0.0.1:8000
```

想切回 ES 版本做对比，只要把 `VALUE_STORE_PROVIDER=es` 并把 ES 起起来即可，MySQL 实现的代码路径完全不会被执行。
