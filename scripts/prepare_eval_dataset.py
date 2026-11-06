"""
从多个数据源生成 RAG 评估数据集。

数据源:
  1. MedQA 执业医师考试题 → 提取问题 (去掉选项，变为开放式问答)
  2. medical.json 种子数据 → 自动生成疾病/药物/症状相关问题

输出: data/eval/rag_eval_dataset.json

用法:
  python scripts/prepare_eval_dataset.py
  python scripts/prepare_eval_dataset.py --size 200
"""
import json
import os
import sys
import random
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "eval")
MEDICAL_JSON = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "medical.json")


def load_medqa_questions(max_count: int = 50) -> list[dict]:
    """从 MedQA 提取开放式问题 (去掉选项)"""
    medqa_path = os.path.join(EVAL_DIR, "medqa_zh.json")
    if not os.path.exists(medqa_path):
        print(f"[WARN] {medqa_path} 不存在，请先运行: python scripts/init_public_datasets.py --dataset medqa")
        return []

    with open(medqa_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 只取 test split，去掉选项变为开放式问答
    test_items = [r for r in raw if r.get("split") == "test"]
    if not test_items:
        test_items = raw

    random.shuffle(test_items)
    questions = []
    for item in test_items[:max_count]:
        q = item["question"]
        # 去掉末尾的"下列哪项正确"等选择题尾巴
        for suffix in ["，最可能的诊断是", "，应首先考虑", "，最佳治疗方案是", "，最可能的原因是"]:
            if suffix in q:
                q = q.split(suffix)[0] + suffix + "什么？"
                break

        answer_text = ""
        if item.get("answer") and item.get("choices"):
            # 将正确选项作为参考答案
            ans_labels = item["answer"]
            choices = item["choices"]
            for choice in choices:
                for label in ans_labels:
                    if choice.startswith(label):
                        answer_text = choice[2:].strip()  # 去掉 "A." 前缀

        questions.append({
            "question": q,
            "reference_answer": answer_text,
            "source": "medqa",
            "category": "clinical",
        })

    return questions


def generate_from_medical_json(max_count: int = 50) -> list[dict]:
    """从 medical.json 自动生成评估问题"""
    if not os.path.exists(MEDICAL_JSON):
        print(f"[WARN] {MEDICAL_JSON} 不存在")
        return []

    # 读取 JSONL 格式
    diseases = []
    with open(MEDICAL_JSON, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    diseases.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    random.shuffle(diseases)
    questions = []

    # 问题模板
    templates = [
        ("symptom", "{name}有哪些症状？", lambda d: "、".join(d.get("symptom", [])[:5])),
        ("cause", "{name}的病因是什么？", lambda d: d.get("cause", "")),
        ("prevent", "{name}如何预防？", lambda d: d.get("prevent", "")),
        ("drug", "{name}常用什么药物治疗？", lambda d: "、".join(d.get("common_drug", [])[:5])),
        ("check", "{name}需要做哪些检查？", lambda d: "、".join(d.get("check", [])[:5])),
        ("department", "{name}应该挂什么科？", lambda d: "、".join(d.get("cure_department", [])[:3])),
        ("diet", "{name}患者饮食上要注意什么？", lambda d: "宜吃: " + "、".join(d.get("recommand_eat", [])[:3])),
    ]

    for disease in diseases:
        if len(questions) >= max_count:
            break

        name = disease.get("name", "")
        if not name or len(name) < 2:
            continue

        # 随机选一个模板
        tpl_type, tpl_question, answer_fn = random.choice(templates)
        ref_answer = answer_fn(disease)
        if not ref_answer or len(ref_answer) < 3:
            continue

        questions.append({
            "question": tpl_question.format(name=name),
            "reference_answer": ref_answer,
            "source": "medical_json",
            "category": tpl_type,
        })

    return questions


def generate_manual_questions() -> list[dict]:
    """手动编写的高质量评估问题 (贴近真实用户场景)"""
    return [
        {
            "question": "高血压患者能吃布洛芬吗？",
            "reference_answer": "高血压患者应慎用布洛芬，因为NSAIDs类药物可能升高血压、影响降压药效果",
            "source": "manual",
            "category": "drug_interaction",
        },
        {
            "question": "糖尿病人可以吃西瓜吗？",
            "reference_answer": "糖尿病患者可以少量食用西瓜，但需注意西瓜GI值较高(72)，建议每次不超过200g",
            "source": "manual",
            "category": "diet",
        },
        {
            "question": "感冒和流感有什么区别？",
            "reference_answer": "普通感冒症状较轻以鼻部症状为主，流感起病急、全身症状重(高热、肌肉酸痛)，由流感病毒引起",
            "source": "manual",
            "category": "differential_diagnosis",
        },
        {
            "question": "孕妇发烧了能吃什么药？",
            "reference_answer": "孕妇发烧首选对乙酰氨基酚(扑热息痛)，禁用布洛芬和阿司匹林，体温超过38.5°C应就医",
            "source": "manual",
            "category": "drug_safety",
        },
        {
            "question": "头孢和酒能一起吗？",
            "reference_answer": "绝对不能。头孢类抗生素与酒精同服会引起双硫仑样反应，可能导致面部潮红、心悸、甚至休克",
            "source": "manual",
            "category": "drug_interaction",
        },
        {
            "question": "小孩反复咳嗽一个月了怎么办？",
            "reference_answer": "儿童慢性咳嗽(>4周)需排除咳嗽变异性哮喘、上气道咳嗽综合征、感染后咳嗽等，建议儿科就诊做肺功能检查",
            "source": "manual",
            "category": "clinical",
        },
        {
            "question": "阿莫西林和头孢有什么区别？",
            "reference_answer": "阿莫西林属于青霉素类，头孢属于头孢菌素类，两者都是β-内酰胺类抗生素但抗菌谱不同，头孢对青霉素过敏者需谨慎",
            "source": "manual",
            "category": "drug_knowledge",
        },
        {
            "question": "胃疼应该做什么检查？",
            "reference_answer": "胃痛建议做胃镜检查(金标准)、幽门螺杆菌检测(C13/C14呼气试验)、腹部超声排除胆胰疾病",
            "source": "manual",
            "category": "check",
        },
        {
            "question": "血压高到多少需要吃药？",
            "reference_answer": "一般收缩压≥140mmHg或舒张压≥90mmHg且生活方式干预3个月无效时需药物治疗，合并糖尿病/肾病者标准更低(≥130/80)",
            "source": "manual",
            "category": "clinical_decision",
        },
        {
            "question": "甲状腺结节需要手术吗？",
            "reference_answer": "多数良性结节无需手术，定期随访即可。需手术的情况：TI-RADS 4类以上、细针穿刺提示恶性、结节>4cm压迫症状明显",
            "source": "manual",
            "category": "clinical_decision",
        },
    ]


def main():
    parser = argparse.ArgumentParser(description="生成 RAG 评估数据集")
    parser.add_argument("--size", type=int, default=100, help="总数据量 (默认 100)")
    parser.add_argument("--medqa-ratio", type=float, default=0.4, help="MedQA 占比 (默认 0.4)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(EVAL_DIR, exist_ok=True)

    medqa_count = int(args.size * args.medqa_ratio)
    medical_count = args.size - medqa_count - 10  # 预留 10 条手动题

    print(f"[INFO] 生成评估数据集: 总量={args.size}, MedQA={medqa_count}, medical.json={medical_count}, 手动=10")

    # 收集各来源数据
    dataset = []
    dataset.extend(generate_manual_questions())
    dataset.extend(load_medqa_questions(max_count=medqa_count))
    dataset.extend(generate_from_medical_json(max_count=medical_count))

    # 截断到目标大小
    if len(dataset) > args.size:
        dataset = dataset[:args.size]

    # 保存
    output_path = os.path.join(EVAL_DIR, "rag_eval_dataset.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    # 统计
    sources = {}
    categories = {}
    for item in dataset:
        sources[item["source"]] = sources.get(item["source"], 0) + 1
        categories[item["category"]] = categories.get(item["category"], 0) + 1

    print(f"\n[OK] 评估数据集已生成: {output_path}")
    print(f"     总量: {len(dataset)} 条")
    print(f"     来源分布: {sources}")
    print(f"     类别分布: {categories}")
    print(f"\n[示例]")
    for item in dataset[:3]:
        print(f"  Q: {item['question'][:50]}...")
        print(f"  A: {item['reference_answer'][:50]}...")
        print()


if __name__ == "__main__":
    main()