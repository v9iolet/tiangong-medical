import io

from minio import Minio
from minio.error import S3Error
from urllib3 import PoolManager, Timeout
from urllib3.util.retry import Retry

from src.core.config import get_settings
from loguru import logger


# 获取配置
settings = get_settings()

# 创建 MinIO 客户端
_minio_client = Minio(
    settings.MINIO_ENDPOINT,
    access_key=settings.MINIO_ACCESS_KEY,
    secret_key=settings.MINIO_SECRET_KEY,
    secure=settings.MINIO_SECURE,
    http_client=PoolManager(timeout=Timeout(connect=3, read=10), retries=Retry(total=1)),
)

# 暴露 MinIO 客户端实例
def get_minio_client() -> Minio:
    """
    FastAPI Depends 注入用。
    直接返回模块级别的 MinIO 客户端实例，不需要每次创建新实例。
    """
    return _minio_client


# 保证桶的存在，如果不存在则创建； 再项目的启动周期中调用
def ensure_bucket_exists() -> None:
    """
    确保桶的存在，如果不存在则创建。
    """
    b = _minio_client.bucket_exists(settings.MINIO_BUCKET)
    logger.info(f"检查桶 {settings.MINIO_BUCKET} 是否存在：{b}")
    if not b:
        _minio_client.make_bucket(bucket_name=settings.MINIO_BUCKET)


# 上传文件到 MinIO 桶； 接受文件流和文件名作为参数
def upload_file(object_name: str, data: bytes,
                content_type: str = "application/octet-stream") -> str:
    """
    上传文件到 MinIO 桶。

    :param object_name: 文件名，用于在 MinIO 桶中唯一标识文件。
    :param data: 文件流数据。
    :param content_type: 文件内容类型，默认值为 "application/octet-stream"。
    :return: 文件名
    """
    # ensure_bucket_exists()   项目启动时调用，确保桶的存在

    _minio_client.put_object(bucket_name=settings.MINIO_BUCKET,
                             object_name=object_name,
                             data=io.BytesIO(data),
                             content_type=content_type,
                             length=len(data))

    return object_name



def download_file(object_name: str) -> bytes:
    """
    从 MinIO 桶下载文件。

    :param object_name: 文件名，用于在 MinIO 桶中唯一标识文件。
    :return: None
    """

    resp = _minio_client.get_object(bucket_name=settings.MINIO_BUCKET,
                             object_name=object_name)

    try:
        return resp.read()
    except Exception as e:
        logger.error(f"下载文件 {object_name} 失败：{e}")
        raise e
    finally:
        resp.close() # 关闭响应流
        resp.release_conn() # 释放连接


def delete_object(object_name: str) -> None:
    """
    删除 MinIO 桶中的文件。
    :param object_name: 文件名，用于在 MinIO 桶中唯一标识文件。
    :return: None
    """

    _minio_client.remove_object(bucket_name=settings.MINIO_BUCKET,
                             object_name=object_name)
    logger.info(f"删除文件 {object_name} 成功")


def object_exists(object_name: str) -> bool:
    """只把不存在当作 False；鉴权和网络错误由调用方处理。"""
    try:
        _minio_client.stat_object(settings.MINIO_BUCKET, object_name)
        return True
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
            return False
        raise
