"""
Render 后端部署脚本

用 Render REST API 创建（或更新）后端的免费 Docker web service。
相比在控制台点，脚本化的好处是环境变量一次性配齐、可重复执行、出错能立刻看到原文。

用法（先加载凭据，再执行）：

    set -a; source .env.cloud.local; set +a
    .venv/bin/python deploy/render_deploy.py            # 创建或更新服务，并触发部署
    .venv/bin/python deploy/render_deploy.py --status    # 只看当前状态与最近一次部署

说明：
- 服务已存在时只更新环境变量并重新部署，不会重复创建。
- 不打印任何密钥内容，只输出服务元信息与部署状态。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

API = "https://api.render.com/v1"
SERVICE_NAME = "ec-bot-api"
BRANCH = "deploy/render-vercel"
REPO = "https://github.com/kanglover/agent"
ROOT_DIR = "projects/ec_bot"

# 需要从本机环境透传到 Render 的应用运行时变量。
# 值里带 sync:false 语义的（凭据类）都从当前 shell 环境读取。
PASSTHROUGH = [
    # 大模型网关
    "MODEL_NAME",
    "LLM_API_KEY",
    "LLM_API_BASE",
    # 元数据库
    "META_DB_HOST",
    "META_DB_PORT",
    "META_DB_USER",
    "META_DB_PASSWORD",
    "META_DB_NAME",
    # 数仓库
    "DW_DB_HOST",
    "DW_DB_PORT",
    "DW_DB_USER",
    "DW_DB_PASSWORD",
    "DW_DB_NAME",
    # 向量库
    "QDRANT_PROVIDER",
    "QDRANT_HOST",
    "QDRANT_PORT",
    "QDRANT_API_KEY",
    "QDRANT_USE_HTTPS",
    # Embedding
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_API_BASE",
    "EMBEDDING_SIZE",
]

# 固定值由脚本给出，避免手工填错
FIXED = {
    "META_DB_POOL_SIZE": "3",
    "DW_DB_POOL_SIZE": "3",
    "META_DB_SSL": "true",
    "DW_DB_SSL": "true",
    "DB_SSL_VERIFY": "false",
    "VALUE_STORE_PROVIDER": "mysql",
    "LOG_FILE_ENABLE": "false",
    "LOG_CONSOLE_ENABLE": "true",
    "PYTHONUNBUFFERED": "1",
}


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _build_env_vars() -> list[dict]:
    """收集要写到 Render 的环境变量，缺一个都直接报错（避免部署完才发现漏配）"""

    vars_map: dict[str, str] = dict(FIXED)

    missing = [k for k in PASSTHROUGH if not os.environ.get(k)]
    if missing:
        print("以下变量在当前环境里没有值，无法继续：")
        for k in missing:
            print(f"  - {k}")
        print("\n请检查 .env.cloud.local 是否填全，并确认已 source。")
        sys.exit(1)

    for key in PASSTHROUGH:
        vars_map[key] = os.environ[key]

    # 前端地址已知时顺手把 CORS 放行配好，省一轮来回
    origin = os.environ.get("FRONTEND_ORIGIN", "").strip()
    if origin:
        vars_map["CORS_ALLOW_ORIGINS"] = origin
    # Vercel 每次推送都会生成新预览域名，用正则一次性覆盖
    vars_map["CORS_ALLOW_ORIGIN_REGEX"] = r"https://.*\.vercel\.app"

    return [{"key": k, "value": v} for k, v in vars_map.items()]


def _find_service(token: str, owner_id: str) -> dict | None:
    resp = requests.get(
        f"{API}/services",
        headers=_headers(token),
        params={"ownerId": owner_id, "limit": 100},
        timeout=60,
    )
    resp.raise_for_status()
    for item in resp.json():
        svc = item.get("service", item)
        if svc.get("name") == SERVICE_NAME:
            return svc
    return None


def _get_owner_id(token: str) -> str:
    resp = requests.get(f"{API}/owners", headers=_headers(token), params={"limit": 10}, timeout=60)
    resp.raise_for_status()
    owners = resp.json()
    if not owners:
        print("该 API Key 下没有任何 workspace")
        sys.exit(1)
    owner = owners[0].get("owner", owners[0])
    return owner["id"]


def _show_status(token: str, service_id: str) -> None:
    resp = requests.get(f"{API}/services/{service_id}", headers=_headers(token), timeout=60)
    resp.raise_for_status()
    svc = resp.json()
    details = svc.get("serviceDetails", {})
    print(f"  名称      : {svc.get('name')}")
    print(f"  类型/计划 : {svc.get('type')} / {details.get('plan')}")
    print(f"  分支      : {svc.get('branch')}")
    print(f"  地址      : {svc.get('serviceDetails', {}).get('url') or svc.get('serviceDetails')}")
    print(f"  状态      : {svc.get('suspended')=}  {svc.get('autoDeploy')=}")

    deploys = requests.get(
        f"{API}/services/{service_id}/deploys",
        headers=_headers(token),
        params={"limit": 3},
        timeout=60,
    )
    if deploys.ok:
        print("\n  最近部署：")
        for item in deploys.json():
            d = item.get("deploy", item)
            commit = (d.get("commit") or {}).get("id", "")[:8]
            print(
                f"    {d.get('id')}  {d.get('status'):<12} commit={commit}  "
                f"{d.get('createdAt')}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true", help="只查询状态，不做改动")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="创建后阻塞等待部署结束，并打印结果",
    )
    args = parser.parse_args()

    token = os.environ.get("RENDER_API_KEY")
    if not token:
        print("缺少 RENDER_API_KEY")
        return 1

    owner_id = _get_owner_id(token)
    existing = _find_service(token, owner_id)

    if args.status:
        if not existing:
            print(f"服务 {SERVICE_NAME} 还不存在")
            return 1
        _show_status(token, existing["id"])
        return 0

    env_vars = _build_env_vars()

    if existing:
        service_id = existing["id"]
        print(f"服务已存在（{service_id}），更新环境变量…")
        resp = requests.put(
            f"{API}/services/{service_id}/env-vars",
            headers=_headers(token),
            json=env_vars,
            timeout=60,
        )
        if not resp.ok:
            print(f"更新环境变量失败 {resp.status_code}: {resp.text[:400]}")
            return 1
        print(f"  已写入 {len(env_vars)} 个环境变量")
        dep = requests.post(
            f"{API}/services/{service_id}/deploys",
            headers=_headers(token),
            json={"clearCache": "do_not_clear"},
            timeout=60,
        )
        if not dep.ok:
            print(f"触发部署失败 {dep.status_code}: {dep.text[:300]}")
            return 1
        deploy_id = dep.json().get("id")
        print(f"  已触发部署 {deploy_id}")
    else:
        payload = {
            "type": "web_service",
            "name": SERVICE_NAME,
            "ownerId": owner_id,
            "repo": REPO,
            "branch": BRANCH,
            "autoDeploy": "yes",
            "envVars": env_vars,
            "serviceDetails": {
                "env": "docker",
                "plan": "free",
                "region": "singapore",
                "healthCheckPath": "/health",
                # 仓库是 monorepo，构建上下文要指向子目录
                "envSpecificDetails": {
                    "dockerfilePath": f"./{ROOT_DIR}/Dockerfile",
                    "dockerContext": f"./{ROOT_DIR}",
                },
            },
        }
        resp = requests.post(f"{API}/services", headers=_headers(token), json=payload, timeout=90)
        if not resp.ok:
            print(f"创建服务失败 {resp.status_code}")
            print(resp.text[:800])
            return 1
        svc = resp.json()
        service_id = svc.get("id") or svc.get("service", {}).get("id")
        print(f"服务创建成功：{service_id}")
        print(f"  已写入 {len(env_vars)} 个环境变量")

    details = requests.get(
        f"{API}/services/{service_id}", headers=_headers(token), timeout=60
    ).json()
    url = (details.get("serviceDetails") or {}).get("url")
    print(f"  访问地址：{url}")
    print(f"  控制台：https://dashboard.render.com/web/{service_id}")

    if args.wait:
        print("\n等待首次部署完成（免费实例首次构建通常 5-10 分钟）…")
        deadline = time.time() + 1500
        last = None
        while time.time() < deadline:
            r = requests.get(
                f"{API}/services/{service_id}/deploys",
                headers=_headers(token),
                params={"limit": 1},
                timeout=60,
            )
            if r.ok and r.json():
                d = r.json()[0].get("deploy", r.json()[0])
                status = d.get("status")
                if status != last:
                    print(f"  [{time.strftime('%H:%M:%S')}] {status}")
                    last = status
                if status in ("live", "build_failed", "canceled", "deploy_failed", "update_failed"):
                    print(f"\n最终状态：{status}")
                    if status != "live":
                        print(f"详情：https://dashboard.render.com/web/{service_id}/deploys/{d.get('id')}")
                    return 0 if status == "live" else 1
            time.sleep(15)
        print("等待超时，请到控制台查看")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
