"""聊天调用上下文：用户、业务会话和患者档案是三个独立概念。"""

from dataclasses import dataclass


def make_thread_id(user_id: str, session_id: str) -> str:
    """Supervisor 和问诊共用的会话线程 ID，不从中反向解析身份。"""
    if not user_id or not session_id:
        raise ValueError("user_id 和 session_id 不能为空")
    return f"{user_id}:{session_id}"


@dataclass
class UserContext:
    user_id: str
    session_id: str
    patient_id: int | None = None

    @property
    def thread_id(self) -> str:
        return make_thread_id(self.user_id, self.session_id)
