import bcrypt

# 对密码进行哈希加密
def hash_password(plain_password: str) -> str:
    """明文密码 → bcrypt 哈希"""
    return bcrypt.hashpw(
        plain_password.encode("utf-8"),
        bcrypt.gensalt(), # 生成盐值
    ).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """校验明文密码是否匹配哈希"""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )