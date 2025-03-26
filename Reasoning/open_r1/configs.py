# coding=utf-8
# Copyright 2025 HuggingFace 팀. 모든 권리 보유.
#
# Apache 라이선스 2.0(이하 "라이선스")에 따라 라이선스가 부여됨;
# 라이선스를 준수하지 않는 한 이 파일을 사용할 수 없음.
# 라이선스의 사본은 다음에서 얻을 수 있음:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 관련 법률에서 요구하거나 서면으로 동의하지 않는 한, 라이선스에 따라 배포되는 소프트웨어는
# 명시적이든 묵시적이든 어떠한 종류의 보증이나 조건 없이 "있는 그대로" 배포됨.
# 라이선스에 따른 특정 언어의 권한 및 제한사항에 대한 자세한 내용은 라이선스를 참조.

# 이 파일은 강화학습 기반 모델 훈련(GRPO, SFT)을 위한 설정 클래스들을 정의합니다.
# trl 라이브러리의 설정을 확장하여 추가 기능을 제공합니다.

from dataclasses import dataclass, field
from typing import Optional

import trl


@dataclass
class GRPOScriptArguments(trl.ScriptArguments):
    """
    GRPO 훈련 스크립트의 인자 클래스.
    다양한 보상 함수 및 관련 설정을 정의합니다.

    Args:
        reward_funcs (`list[str]`):
            보상 함수 목록. 가능한 값: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', 'tag_count', 'code', 'ioi_code', 'code_format'.
        cosine_min_value_wrong (`float`):
            잘못된 답변에 대한 코사인 스케일링의 최소 보상.
        cosine_max_value_wrong (`float`):
            잘못된 답변에 대한 코사인 스케일링의 최대 보상.
        cosine_min_value_correct (`float`):
            올바른 답변에 대한 코사인 스케일링의 최소 보상.
        cosine_max_value_correct (`float`):
            올바른 답변에 대한 코사인 스케일링의 최대 보상.
        cosine_max_len (`int`):
            코사인 스케일링의 최대 길이.
        code_language (`str`):
            코드 형식 보상을 위한 언어.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "보상 함수 목록. 가능한 값: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "잘못된 답변에 대한 최소 보상"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "잘못된 답변에 대한 최대 보상"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "올바른 답변에 대한 최소 보상"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "올바른 답변에 대한 최대 보상"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "스케일링을 위한 최대 길이"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "반복 페널티 보상을 위한 n-gram 수"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "반복 페널티 보상에 대한 최대 (음수) 페널티"},
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "코드 형식 보상을 위한 언어. E2B 지원 언어 기반 https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash", "cpp"],
        },
    )
    code_eval_test_batch_size: int = field(
        default=1,
        metadata={
            "help": "각 생성에 대해 이만큼의 테스트 케이스를 병렬로 평가한 다음, 실패한 케이스가 있는지 확인(0점): 있으면 평가 중지; 그렇지 않으면 다음 테스트 케이스 배치로 계속 진행. 평가 서버에 과부하를 방지하고 잘못된 솔루션에 대한 시간을 절약하는 데 유용"
        },
    )
