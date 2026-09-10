"""
ColumnInfo 映射器

负责在字段元数据业务实体和 ORM 模型之间做转换，使字段入库过程保持
“业务实体 -> Mapper -> ORM 模型”的清晰分层
"""

from dataclasses import asdict

from app.entities.column_info import ColumnInfo
from app.models.column_info import ColumnInfoMySQL

class ColumnInfoMapper:

    @staticmethod
    def to_entity(column_info_mysql: ColumnInfoMySQL) -> ColumnInfo:
        return ColumnInfo(
            id=column_info_mysql.id,
            name=column_info_mysql.name or "",
            type=column_info_mysql.type or "",
            role=column_info_mysql.role or "",
            examples=column_info_mysql.examples or [],
            description=column_info_mysql.description or "",
            alias=column_info_mysql.alias or [],
            table_id=column_info_mysql.table_id or "",
        )

    @staticmethod
    def to_model(column_info: ColumnInfo) -> ColumnInfoMySQL:
        """把字段业务实体转换成 ORM 模型用于持久化"""
        return ColumnInfoMySQL(**asdict(column_info))