from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

# 定义常用的错误码，错误消息
ERROR_CODE_400 = 400
ERROR_MSG_400 = "请求参数错误"

ERROR_CODE_404 = 404
ERROR_MSG_404 = "资源不存在"

ERROR_CODE_500 = 500
ERROR_MSG_500 = "服务器内部错误"

ERR_PERM_NOT_FOUND = 40003
ERR_PERM_NOT_FOUND_MSG = "权限不存在"


class BizException(Exception):
    """业务异常"""
    def __init__(self, code: int = 400, message: str = "业务异常"):
        self.code = code
        self.message = message

def register_exception_handlers(app: FastAPI) -> None:
    
    @app.exception_handler(BizException)
    async def biz_exception_handler(request: Request, exc: BizException):
        return JSONResponse(
            status_code=200,
            content={"code": exc.code, "message": exc.message, "data": None},
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"code": 500, "message": "服务器内部错误", "data": None},
        )