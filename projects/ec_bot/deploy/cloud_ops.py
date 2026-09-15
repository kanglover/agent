"""
云端部署辅助脚本（在本地执行，不进容器）

把「接通四个云服务 + 初始化数据库」这段最容易反复试错的流程固化成一个命令，
避免手工敲 mysql 客户端（本机也不一定有装）。

用法（先加载凭据文件，再执行子命令）：

    set -a; source .env.cloud.local; set +a

    .venv/bin/python deploy/cloud_ops.py check      # 四路连通性 + 向量维度校验
    .venv/bin/python deploy/cloud_ops.py setup-db    # 建库、导表结构、导数据、校验

说明：
- 直接复用应用自己的 MySQLClientManager 等组件，保证脚本验证的行为与线上运行时完全一致
  （包括 TLS 参数拼接方式），避免「脚本能跑、服务连不上」这类割裂问题。
- setup-db 会执行 DROP TABLE，属于重建语义，只适合演示库；库里已有真实数据时不要跑。
"""

import argparse
import asyncio
import re
import sys
from dataclasses import replace
from pathlib import Path

from sqlalchemy import text

# 脚本放在 deploy/ 下，需要把项目根目录加进模块搜索路径才能 import app
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.clients.embedding_client_manager import embedding_client_manager  # noqa: E402
from app.clients.mysql_client_manager import MySQLClientManager  # noqa: E402
from app.clients.qdrant_client_manager import qdrant_client_manager  # noqa: E402
from app.conf.app_config import DBConfig, app_config  # noqa: E402

# 待导入的 SQL 与目标库的对应关系（路径基于项目根目录，避免受当前工作目录影响）
SQL_TARGETS = {
    "meta": PROJECT_ROOT / "docker/mysql/remote/meta_schema.sql",
    "dw": PROJECT_ROOT / "docker/mysql/remote/dw_schema.sql",
}

# 校验数据量时抽查的表
EXPECTED_TABLES = {
    "meta": ["table_info", "column_info", "metric_info", "column_metric"],
    "dw": ["dim_region", "dim_customer", "dim_product", "dim_date", "fact_order"],
}

# Aiven 等托管实例自带这个库，建库前先连它
BOOTSTRAP_DATABASE = "defaultdb"


def _ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m {msg}")


def _fail(msg: str) -> None:
    print(f"  \033[31m✗\033[0m {msg}")


def _split_sql(sql_text: str) -> list[str]:
    """
    按分号切分 SQL 语句

    不能用简单的 split(';')：字符串字面量里的分号必须保留。
    这里逐字符扫描，跟踪单引号 / 双引号 / 反引号状态，并跳过 -- 与 /* */ 注释。
    """

    statements: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    i = 0
    length = len(sql_text)

    while i < length:
        ch = sql_text[i]

        if quote:
            buf.append(ch)
            if ch == "\\" and quote in ("'", '"'):
                # 转义字符，连同下一个字符一起吃掉
                if i + 1 < length:
                    buf.append(sql_text[i + 1])
                    i += 2
                    continue
            elif ch == quote:
                # 连续两个引号表示一个转义引号，不算结束
                if i + 1 < length and sql_text[i + 1] == quote:
                    buf.append(sql_text[i + 1])
                    i += 2
                    continue
                quote = None
            i += 1
            continue

        # 非引号状态：先看注释
        if ch == "-" and sql_text.startswith("--", i):
            end = sql_text.find("\n", i)
            i = length if end == -1 else end + 1
            continue
        if ch == "/" and sql_text.startswith("/*", i):
            end = sql_text.find("*/", i)
            i = length if end == -1 else end + 2
            continue

        if ch in ("'", '"', "`"):
            quote = ch
            buf.append(ch)
            i += 1
            continue

        if ch == ";":
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
            i += 1
            continue

        buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


def _bootstrap_manager(cfg: DBConfig) -> MySQLClientManager:
    """构造一个连到 bootstrap 库的 manager，用于创建目标库"""

    return MySQLClientManager(replace(cfg, database=BOOTSTRAP_DATABASE))


async def _ensure_database(manager: MySQLClientManager, dbname: str) -> bool:
    """库不存在则创建，返回是否是本次新建的"""

    engine = manager.engine
    assert engine is not None
    async with engine.begin() as conn:
        result = await conn.execute(
            text(
                "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA "
                "WHERE SCHEMA_NAME = :name"
            ),
            {"name": dbname},
        )
        exists = result.first() is not None
        if not exists:
            # 库名来自本地配置而非用户输入，这里仍做一次白名单式校验
            if not re.fullmatch(r"[A-Za-z0-9_]+", dbname):
                raise ValueError(f"非法库名：{dbname}")
            await conn.execute(text(f"CREATE DATABASE `{dbname}` CHARACTER SET utf8mb4"))
    return not exists


async def _import_sql(manager: MySQLClientManager, sql_path: Path) -> int:
    """把 SQL 文件逐条执行，返回执行的语句数"""

    statements = _split_sql(sql_path.read_text(encoding="utf-8"))
    engine = manager.engine
    assert engine is not None
    async with engine.begin() as conn:
        for stmt in statements:
            await conn.execute(text(stmt))
    return len(statements)


async def _count_rows(manager: MySQLClientManager, table: str) -> int:
    engine = manager.engine
    assert engine is not None
    async with engine.begin() as conn:
        result = await conn.execute(text(f"SELECT COUNT(*) FROM `{table}`"))
        return int(result.scalar_one())


async def cmd_check() -> int:
    """四路连通性检查，并校验 embedding 真实维度与配置是否一致"""

    failures = 0

    print("\n[1/4] MySQL（Aiven 等托管实例）")
    managers = {
        "meta": MySQLClientManager(app_config.db_meta),
        "dw": MySQLClientManager(app_config.db_dw),
    }
    for name, manager in managers.items():
        manager.init()
        try:
            engine = manager.engine
            assert engine is not None
            async with engine.begin() as conn:
                version = (await conn.execute(text("SELECT VERSION()"))).scalar_one()
                tls = (await conn.execute(text("SHOW STATUS LIKE 'Ssl_cipher'"))).first()
            cipher = tls[1] if tls else ""
            _ok(
                f"{name}: {manager.config.host}:{manager.config.port}"
                f"/{manager.config.database}  版本 {version}"
                f"  TLS={'已启用 ' + cipher if cipher else '未启用'}"
            )
            if not cipher:
                print("     \033[33m! 连接未走 TLS，托管实例通常强制 TLS，请检查 *_DB_SSL 配置\033[0m")
        except Exception as exc:  # noqa: BLE001 连通性检查需要兜住所有异常
            _fail(f"{name}: {type(exc).__name__}: {exc}")
            failures += 1
        finally:
            await manager.close()

    print("\n[2/4] Qdrant")
    try:
        qdrant_client_manager.init()
        client = qdrant_client_manager.client
        assert client is not None
        collections = await client.get_collections()
        names = [c.name for c in collections.collections]
        _ok(
            f"{app_config.qdrant.host}:{app_config.qdrant.port}"
            f"  已有集合 {len(names)} 个：{names or '（空，构建时会自动创建）'}"
        )
    except Exception as exc:  # noqa: BLE001
        _fail(f"{type(exc).__name__}: {exc}")
        failures += 1
    finally:
        await qdrant_client_manager.close()

    print("\n[3/4] Embedding")
    try:
        embedding_client_manager.init()
        client = embedding_client_manager.client
        vector = await client.aembed_query("华东地区的销售额")
        actual = len(vector)
        expected = app_config.qdrant.embedding_size
        _ok(f"{app_config.embedding.provider} / {app_config.embedding.model}  返回维度 {actual}")
        if actual == expected:
            _ok(f"EMBEDDING_SIZE={expected} 与模型真实维度一致")
        else:
            _fail(
                f"维度不匹配：模型实际返回 {actual}，但 EMBEDDING_SIZE={expected}。"
                f"必须把 EMBEDDING_SIZE 改成 {actual} 再构建索引"
            )
            failures += 1
    except Exception as exc:  # noqa: BLE001
        _fail(f"{type(exc).__name__}: {exc}")
        failures += 1

    print("\n[4/4] LLM")
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=app_config.llm.api_key, base_url=app_config.llm.base_url
        )
        resp = await client.chat.completions.create(
            model=app_config.llm.model_name,
            messages=[{"role": "user", "content": "回复两个字：可用"}],
            max_tokens=16,
        )
        _ok(
            f"{app_config.llm.model_name} @ {app_config.llm.base_url}"
            f"  返回：{(resp.choices[0].message.content or '').strip()[:20]}"
        )
    except Exception as exc:  # noqa: BLE001
        _fail(f"{type(exc).__name__}: {exc}")
        failures += 1

    print()
    if failures:
        print(f"检查未通过，共 {failures} 项失败。修好后再继续。")
    else:
        print("四路全部就绪。")
    return failures


async def cmd_setup_db(force: bool) -> int:
    """建库 + 导表结构 + 导数据 + 校验"""

    # 先建库：目标库可能还不存在，所以从 defaultdb 连进去
    print("\n[1/3] 创建数据库")
    bootstrap = {
        "meta": _bootstrap_manager(app_config.db_meta),
        "dw": _bootstrap_manager(app_config.db_dw),
    }
    for name, manager in bootstrap.items():
        manager.init()
        try:
            created = await _ensure_database(manager, name)
            _ok(f"{name}：{'本次新建' if created else '已存在，跳过'}")
        finally:
            await manager.close()

    # 再导数据
    print("\n[2/3] 导入表结构与初始数据")
    targets = {
        "meta": MySQLClientManager(app_config.db_meta),
        "dw": MySQLClientManager(app_config.db_dw),
    }
    for name, manager in targets.items():
        manager.init()
        try:
            sql_path = SQL_TARGETS[name]
            statements = await _import_sql(manager, sql_path)
            _ok(f"{name}：执行 {statements} 条语句（{sql_path}）")
        except Exception as exc:  # noqa: BLE001
            if not force:
                _fail(
                    f"{name}：{type(exc).__name__}: {exc}\n"
                    f"     若确认要覆盖已有数据，加 --force 重试"
                )
                return 1
            raise
        finally:
            await manager.close()

    # 最后校验
    print("\n[3/3] 校验结果")
    checks = {
        "meta": MySQLClientManager(app_config.db_meta),
        "dw": MySQLClientManager(app_config.db_dw),
    }
    total = 0
    for name, manager in checks.items():
        manager.init()
        try:
            for table in EXPECTED_TABLES[name]:
                count = await _count_rows(manager, table)
                _ok(f"{name}.{table}: {count} 行")
                total += count
        finally:
            await manager.close()

    print(f"\n完成，共 {total} 行数据。")
    print("下一步：构建云端索引（写入 Qdrant 集合与 MySQL 取值索引表）")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="ec_bot 云端部署辅助脚本")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="检查四个云服务的连通性")
    setup = sub.add_parser("setup-db", help="建库、导入表结构与数据")
    setup.add_argument(
        "--force",
        action="store_true",
        help="导入失败时也继续（SQL 内含 DROP TABLE，会清掉同名表）",
    )

    args = parser.parse_args()

    if args.command == "check":
        return asyncio.run(cmd_check())
    if args.command == "setup-db":
        return asyncio.run(cmd_setup_db(args.force))
    return 1


if __name__ == "__main__":
    sys.exit(main())
