"""
知识 Agent 评估指标定义
包含：RAG Triad 三指标 + 医学领域自定义指标 + 参考答案一致性
"""
import math
import numpy as np
from trulens.core.metric import Metric
from trulens.core.metric.selector import Selector
from trulens.feedback import GroundTruthAgreement
from trulens.providers.litellm import LiteLLM

ANSWER_RELEVANCE = "答案相关性"
CONTEXT_RELEVANCE = "上下文相关性"
GROUNDEDNESS = "有据性"
HARMFULNESS = "有害性"
GROUND_TRUTH = "参考答案一致性"

ALL_METRIC_NAMES = [ANSWER_RELEVANCE, CONTEXT_RELEVANCE, GROUNDEDNESS, HARMFULNESS, GROUND_TRUTH]


class FiniteMetric(Metric):
    """在 TruLens 写入评分 span 前拒绝非法值，避免 PostgreSQL JSON 丢失整批 span。"""

    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)
        score = result[0] if isinstance(result, tuple) else result
        try:
            value = float(score)
        except (TypeError, ValueError):
            raise ValueError("invalid_metric_score") from None
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("invalid_metric_score")
        return (value, *result[1:]) if isinstance(result, tuple) else value


def mean_context_relevance(scores: list[float]) -> float:
    """零召回记为 0；避免 np.mean([]) 产生 PostgreSQL JSON 不接受的 NaN。"""
    if any(not math.isfinite(float(score)) or not 0 <= float(score) <= 1 for score in scores):
        raise ValueError("invalid_context_score")
    return float(np.mean(scores)) if len(scores) else 0.0


def build_rag_triad_metrics(provider: LiteLLM) -> list[Metric]:
    """RAG Triad 三大核心指标"""

    m_answer_relevance = FiniteMetric(
        implementation=provider.relevance_with_cot_reasons,
        name=ANSWER_RELEVANCE,
        selectors={
            "prompt": Selector.select_record_input(),
            "response": Selector.select_record_output(),
        },
    )

    m_context_relevance = FiniteMetric(
        implementation=provider.context_relevance_with_cot_reasons,
        name=CONTEXT_RELEVANCE,
        selectors={
            "question": Selector.select_record_input(),
            "context": Selector.select_context(collect_list=False),
        },
        agg=mean_context_relevance,
    )

    m_groundedness = FiniteMetric(
        implementation=provider.groundedness_measure_with_cot_reasons,
        name=GROUNDEDNESS,
        selectors={
            "source": Selector.select_context(collect_list=True),
            "statement": Selector.select_record_output(),
        },
    )

    return [m_answer_relevance, m_context_relevance, m_groundedness]


def build_medical_metrics(provider: LiteLLM) -> list[Metric]:
    """医学领域自定义指标"""

    # harmfulness 只看回答文本本身，分数越高越有害
    m_harmfulness = FiniteMetric(
        implementation=provider.harmfulness_with_cot_reasons,
        name=HARMFULNESS,
        higher_is_better=False,
        selectors={
            "text": Selector.select_record_output(),
        },
    )

    return [m_harmfulness]


def build_ground_truth_metric(provider: LiteLLM, golden_set: list[dict]) -> Metric:
    """
    参考答案一致性：按问题原文匹配评估集中的 reference_answer。
    golden_set: [{"query": 问题, "expected_response": 参考答案}, ...]；匹配不到参考答案的记录得分为 NaN。
    """
    agreement = GroundTruthAgreement(golden_set, provider=provider)
    return FiniteMetric(
        implementation=agreement.agreement_measure,
        name=GROUND_TRUTH,
        selectors={
            "prompt": Selector.select_record_input(),
            "response": Selector.select_record_output(),
        },
    )


def build_all_metrics(provider: LiteLLM) -> list[Metric]:
    """全部指标（RAG Triad + 医学领域）"""
    return build_rag_triad_metrics(provider) + build_medical_metrics(provider)


def build_metrics(
    provider: LiteLLM,
    names: list[str] | None = None,
    golden_set: list[dict] | None = None,
) -> list[Metric]:
    """按名称挑选指标；没有参考答案时自动跳过参考答案一致性。"""
    metrics = build_all_metrics(provider)
    if golden_set:
        metrics.append(build_ground_truth_metric(provider, golden_set))
    if names is None:
        return metrics
    return [m for m in metrics if m.name in names]
