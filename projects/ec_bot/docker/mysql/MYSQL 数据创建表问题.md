
## MySQL 初始化问题排查与修复
- 问题：docker compose 启动后 dw/meta 库和表未自动创建，admin 用户 SELECT 报权限拒绝。
- 根因：`mysql_data` 卷在 8/28 首次启动时创建（当时 `./mysql` 下还没有 SQL 文件）；`/docker-entrypoint-initdb.d` 脚本只在数据卷首次初始化（为空）时执行，之后重启/重挂载都不会重跑。admin 由 MYSQL_USER 创建但未设 MYSQL_DATABASE，零权限；SQL 里的 GRANT 从未执行。
- 修复：`docker compose stop mysql && docker compose rm -f mysql && docker volume rm docker_mysql_data && docker compose up -d mysql`，重新初始化后 dw(5表)、meta(4表) 建好，admin 拿到 dw/meta 的 ALL PRIVILEGES，验证查询通过。
- 经验：以后修改 docker/mysql 下的 SQL 后，必须删卷重建才会生效（数据会清空，注意备份）。
