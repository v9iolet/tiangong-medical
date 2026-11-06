"""TruLens 评分模型的 OpenAI 兼容接口适配。"""
from typing import Any

from litellm.exceptions import BadRequestError
from loguru import logger
from pydantic import PrivateAttr
from trulens.providers.litellm import LiteLLM


class CompatibleLiteLLM(LiteLLM):
    """接口明确拒绝结构化输出时使用 TruLens 自带的文本评分解析。"""

    _response_format_supported: bool = PrivateAttr(default=True)

    def _create_chat_completion(
        self, prompt=None, messages=None, response_format=None, **kwargs: Any,
    ) -> str:
        response_format = response_format or self.completion_args.get("response_format")
        if not self._response_format_supported:
            response_format = None
        try:
            return super()._create_chat_completion(
                prompt=prompt, messages=messages, response_format=response_format, **kwargs,
            )
        except BadRequestError as exc:
            message = str(exc).lower()
            unsupported = "response_format" in message and any(
                marker in message for marker in (
                    "unavailable", "unsupported", "not supported", "does not support",
                )
            )
            if response_format is None or not unsupported:
                raise
            # 在 Endpoint 的退避重试之前处理能力差异，避免重复发送必定失败的请求。
            self._response_format_supported = False
            self.completion_args.pop("response_format", None)
            logger.info("评估接口不支持 response_format，后续评分使用文本输出")
            return super()._create_chat_completion(
                prompt=prompt, messages=messages, **kwargs,
            )
