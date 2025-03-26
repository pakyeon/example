"""LightEval을 위한 사용자 정의 평가 태스크."""

import random

from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    IndicesExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


# 프롬프트 템플릿은 다음에서 채택됨:
# - simple-evals: https://github.com/openai/simple-evals/blob/6e84f4e2aed6b60f6a0c7b8f06bbbf4bfde72e58/math_eval.py#L17
# - Llama 3: https://huggingface.co/datasets/meta-llama/Llama-3.2-1B-Instruct-evals/viewer/Llama-3.2-1B-Instruct-evals__math__details?views%5B%5D=llama_32_1b_instruct_evals__math__details
# math-verify가 올바르게 작동하려면 최종 답변이 박스 안에 있어야 함을 유의하세요
MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()

# simple-evals에서 가져온 프롬프트 템플릿: https://github.com/openai/simple-evals/blob/83ed7640a7d9cd26849bcb3340125002ef14abbe/common.py#L14
GPQA_QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

# LaTeX 형식의 정답을 추출하여 평가하는 메트릭 정의
latex_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(LatexExtractionConfig(),),
    # 다른 정규식을 시도하기 전에 먼저 boxed 형식에 매칭
    pred_extraction_target=(
        ExprExtractionConfig(),
        LatexExtractionConfig(boxed_match_priority=0),
    ),
    aggregation_function=max,
)

# 수식 형식의 정답을 추출하여 평가하는 메트릭 정의
expr_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(ExprExtractionConfig(),),
    # 다른 정규식을 시도하기 전에 먼저 boxed 형식에 매칭
    pred_extraction_target=(
        ExprExtractionConfig(),
        LatexExtractionConfig(boxed_match_priority=0),
    ),
    aggregation_function=max,
)

# GPQA(General Physics Question Answering) 평가를 위한 메트릭 정의
gpqa_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    gold_extraction_target=[
        IndicesExtractionConfig(prefix_for_extraction="NativeLetters")
    ],
    pred_extraction_target=[
        IndicesExtractionConfig(prefix_for_extraction="NativeLetters")
    ],
    precision=5,
)


# 수학 문제 프롬프트 생성 함수
# 입력한 문제를 형식화된 쿼리로 변환하는 함수
def math_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["problem"]),
        choices=[line["solution"]],
        gold_index=0,
    )


# AIME(American Invitational Mathematics Examination) 문제 프롬프트 생성 함수
# AIME 문제 형식에 맞게 프롬프트를 생성
def aime_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["problem"]),
        choices=[line["answer"]],
        gold_index=0,
    )


# GPQA 문제 프롬프트 생성 함수
# 물리학 다중 선택 문제를 형식화된 쿼리로 변환하는 함수
def gpqa_prompt_fn(line, task_name: str = None):
    gold_index = random.randint(0, 3)
    choices = [
        line["Incorrect Answer 1"],
        line["Incorrect Answer 2"],
        line["Incorrect Answer 3"],
    ]
    choices.insert(gold_index, line["Correct Answer"])
    query = GPQA_QUERY_TEMPLATE.format(
        A=choices[0],
        B=choices[1],
        C=choices[2],
        D=choices[3],
        Question=line["Question"],
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C", "D"],
        gold_index=gold_index,
        instruction=query,
    )


# 태스크 정의 시작
# AIME 2024 평가 태스크 설정
aime24 = LightevalTaskConfig(
    name="aime24",
    suite=["custom"],
    prompt_function=aime_prompt_fn,
    hf_repo="HuggingFaceH4/aime_2024",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[expr_gold_metric],
    version=1,
)

# AIME 2025 평가 태스크 설정
aime25 = LightevalTaskConfig(
    name="aime25",
    suite=["custom"],
    prompt_function=aime_prompt_fn,
    hf_repo="yentinglin/aime_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[expr_gold_metric],
    version=1,
)

# MATH-500 평가 태스크 설정
math_500 = LightevalTaskConfig(
    name="math_500",
    suite=["custom"],
    prompt_function=math_prompt_fn,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[latex_gold_metric],
    version=1,
)

# GPQA 다이아몬드 레벨 평가 태스크 설정
gpqa_diamond = LightevalTaskConfig(
    name="gpqa:diamond",
    suite=["custom"],
    prompt_function=gpqa_prompt_fn,
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_diamond",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,  # R1과 같은 추론 모델에 필요함
    metric=[gpqa_metric],
    stop_sequence=[],  # 정지 시퀀스 없음, eos 토큰 사용
    trust_dataset=True,
    version=1,
)


# 정의된 태스크를 테이블에 추가
TASKS_TABLE = []
TASKS_TABLE.append(aime24)
TASKS_TABLE.append(aime25)
TASKS_TABLE.append(math_500)
TASKS_TABLE.append(gpqa_diamond)

# 모듈 로직
if __name__ == "__main__":
    print([t["name"] for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
