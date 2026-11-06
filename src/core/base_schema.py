from typing import TypeVar, Generic, Optional
from pydantic import BaseModel

T = TypeVar("T")


class ResponseSchema(BaseModel, Generic[T]):
    code: int = 200
    message: str = "success"
    data: Optional[T] = None

class PageResult(BaseModel, Generic[T]):
    """分页结果包装"""
    items: list[T] = []
    total: int = 0
    page: int = 1
    page_size: int = 20

