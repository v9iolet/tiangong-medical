import os

# LiteLLM 导入时默认会联网拉取模型价格表，网络不通时会卡住并可能触发循环导入；评估只需本地价格表
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
