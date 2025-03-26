"""GRPO 학습을 위한 보상 함수들.

이 모듈은 GRPO(Generative Reinforcement Learning with Preference Optimization) 학습에
사용되는 다양한 보상 함수들을 정의합니다. 이 함수들은 모델의 출력을 평가하고
정답과의 일치도, 형식 준수, 추론 과정의 품질 등 다양한 측면에 기반하여 보상을 계산합니다.
"""

import asyncio
import json
import math
import re
from typing import Dict

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from .utils import is_e2b_available


if is_e2b_available():
    from dotenv import load_dotenv
    from e2b_code_interpreter import (
        AsyncSandbox,
    )  # AI가 생성한 코드를 클라우드 샌드박스 환경에서 안전하게 실행시킬 수 있게 함

    load_dotenv()
else:
    AsyncSandbox = None


def accuracy_reward(completions, solution, **kwargs):
    """정확도 기반 보상 함수로, 모델의 답변이 정답과 일치하는지 확인합니다.

    수학 문제에 특화되어 있으며, LaTeX 형식으로 된 답변을 파싱해 정답과 비교합니다.
    정답과 일치하면 1.0, 그렇지 않으면 0.0의 보상을 반환합니다.

    Args:
        completions: 모델이 생성한 답변 목록
        solution: 정답 목록

    Returns:
        float: 각 답변에 대한 정확도 보상 점수 목록 (0.0 또는 1.0)
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # 답변이 올바른 LaTeX 형식으로 제공되어야 함 (형식이 잘못된 연산자 없이)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # boxed가 먼저 시도되도록 함
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # 내용이 정답과 같으면 1, 아니면 0의 보상
            try:
                reward = float(verify(answer_parsed, gold_parsed))
            except Exception as e:
                print(f"검증 실패: {e}, 답변: {answer_parsed}, 정답: {gold_parsed}")
                reward = 0.0
        else:
            # 정답 솔루션이 파싱 불가능하면 이 예제를 건너뛰기 위해 1의 보상을 줌
            reward = 1.0
            print("정답 솔루션 파싱 실패: ", sol)
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """형식 준수 보상 함수로, 답변이 지정된 태그 구조를 따르는지 확인합니다.

    추론 과정은 <think></think> 태그 안에, 최종 답변은 <answer></answer> 태그 안에
    있어야 합니다. 정규 표현식을 사용해 이 형식을 준수하는지 검사합니다.

    Args:
        completions: 모델이 생성한 답변 목록

    Returns:
        float: 각 답변에 대한 형식 보상 점수 목록 (0.0 또는 1.0)
    """
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [
        re.match(pattern, content, re.DOTALL | re.MULTILINE)
        for content in completion_contents
    ]
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, **kwargs) -> list[float]:
    """태그 개수 보상 함수로, format_reward()와 관련된 태그를 올바르게 사용했는지 확인합니다.

    <think>, </think>, <answer>, </answer> 태그가 각각 정확히 한 번씩 올바른 위치에
    사용되었는지 확인하고, 각 태그별로 0.25점씩 부분 점수를 부여합니다.

    Args:
        completions: 모델이 생성한 답변 목록

    Returns:
        float: 각 답변에 대한 태그 사용 보상 점수 목록 (0.0~1.0 사이의 값)
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""단계적 추론 과정 보상 함수로, 명확한 단계별 추론이 있는지 확인합니다.

    'Step N:', 번호 목록('1.', '2.' 등), 글머리 기호('-', '*'),
    혹은 전환 단어('First,', 'Next,' 등)를 사용한 단계적 추론 과정이 있는지 확인합니다.
    최소 3단계 이상의 추론이 있으면 1.0, 그렇지 않으면 단계 수에 비례한 부분 점수를 부여합니다.

    정규식 패턴:
        Step \d+: - "Step 1:", "Step 2:" 등과 일치
        ^\d+\. - 행 시작 부분의 번호 목록("1.", "2." 등)과 일치
        \n- - 하이픈으로 된 글머리 기호와 일치
        \n\* - 별표로 된 글머리 기호와 일치
        First,|Second,|Next,|Finally, - 전환 단어와 일치

    Args:
        completions: 모델이 생성한 답변 목록

    Returns:
        float: 각 답변에 대한 추론 단계 보상 점수 목록 (0.0~1.0 사이의 값)
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # 매직 넘버 3은 3단계 이상의 추론을 장려하기 위함, 그렇지 않으면 부분 보상
    return [min(1.0, count / 3) for count in matches]


def len_reward(
    completions: list[Dict[str, str]], solution: list[str], **kwargs
) -> float:
    """답변 길이 기반 보상 함수로, 답변의 효율성을 장려하고 불필요한 장황함을 억제합니다.

    Kimi 1.5 기술 보고서에서 가져온 방법으로, 정확한 답변은 짧을수록 높은 보상을,
    부정확한 답변은 길이와 무관하게 낮은 보상을 받습니다.

    Args:
        completions: 모델이 생성한 답변 목록
        solution: 정답 목록

    Returns:
        float: 각 답변에 대한 길이 기반 보상 점수 목록
        - 정확한 답변: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - 부정확한 답변: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # 먼저 답변의 정확성 확인
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # 파싱할 수 없는 예제는 건너뜀
            correctness.append(True)  # 패널티를 피하기 위해 정확한 것으로 처리
            print("정답 솔루션 파싱 실패: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # 길이 계산
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # 모든 응답의 길이가 같으면 0 보상 반환
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    """코사인 스케일링 보상 함수를 생성하는 팩토리 함수입니다.

    답변의 정확성과 길이를 모두 고려하여 코사인 함수로 보상을 조정합니다.
    정확한 답변은 짧을수록 높은 보상을, 부정확한 답변은 길수록 덜 엄격한 패널티를 받습니다.

    Args:
        min_value_wrong: 부정확한 답변의 최소 보상값
        max_value_wrong: 부정확한 답변의 최대 보상값
        min_value_correct: 정확한 답변의 최소 보상값
        max_value_correct: 정확한 답변의 최대 보상값
        max_len: 길이 스케일링을 위한 최대 길이

    Returns:
        function: 파라미터가 적용된 코사인 스케일링 보상 함수
    """

    def cosine_scaled_reward(completions, solution, **kwargs):
        """코사인 함수를 사용해 답변 길이에 따라 보상을 스케일링합니다.

        정확한 답변은 짧을수록 더 높은 보상을 받고,
        부정확한 답변은 길수록 덜 엄격한 패널티를 받습니다.

        Args:
            completions: 모델이 생성한 답변 목록
            solution: 정답 목록

        Returns:
            float: 각 답변에 대한 코사인 스케일링 보상 점수 목록
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # 파싱할 수 없는 예제는 건너뜀
                print("정답 솔루션 파싱 실패: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # 길이에 따른 코사인 스케일링 적용
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # 부정확한 답변은 최소/최대값을 교체
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """반복 패널티 보상 함수를 생성하는 팩토리 함수입니다.

    텍스트 내 N-gram 반복을 감지하고 패널티를 부여합니다.
    중복된 표현이나 반복적인 내용을 억제하기 위한 함수입니다.

    https://arxiv.org/abs/2502.03373 의 부록 C.2에 설명된 N-gram 반복 패널티를 계산합니다.
    참고 구현: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
        ngram_size: N-gram의 크기 (단어 개수)
        max_penalty: 최대 (음수)패널티 값

    Returns:
        function: 파라미터가 적용된 반복 패널티 보상 함수
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty}는 양수가 되어서는 안 됩니다")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """반복 패널티 보상 함수로, 텍스트 내 N-gram 반복에 패널티를 부여합니다.

        답변 내 유니크한 N-gram의 비율을 계산하여 반복이 많을수록 더 큰 패널티를 부여합니다.
        참고 구현: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: 모델이 생성한 답변 목록

        Returns:
            float: 각 답변에 대한 반복 패널티 보상 점수 목록 (0.0 이하의 값)
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def extract_code(completion: str) -> str:
    """답변에서 Python 코드 블록을 추출합니다.

    ```python``` 태그로 둘러싸인 코드를 찾아 마지막 코드 블록을 반환합니다.

    Args:
        completion: 모델이 생성한 답변

    Returns:
        str: 추출된 Python 코드 또는 빈 문자열
    """
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def code_reward(completions, **kwargs) -> list[float]:
    """코드 평가 보상 함수로, E2B 코드 인터프리터를 사용해 코드의 정확성을 평가합니다.

    데이터셋에 포함된 테스트 케이스를 사용해 생성된 코드를 실행하고 정확도를 측정합니다.
    E2B 샌드박스를 사용하여 코드를 안전하게 실행합니다.

    데이터셋에 테스트 케이스가 포함된 'verification_info' 열이 있다고 가정합니다.

    Args:
        completions: 모델이 생성한 답변 목록
        **kwargs: verification_info를 포함한 추가 인자

    Returns:
        float: 각 코드에 대한 성공률 보상 점수 목록 (0.0~1.0 사이의 값)
    """
    if not is_e2b_available():
        raise ImportError(
            "이 보상 함수에 필요한 E2B를 사용할 수 없습니다. "
            "`pip install e2b-code-interpreter`로 E2B를 설치하고 `.env` 파일에 API 키를 추가하세요."
        )

    # TODO: E2B에서 다른 언어 지원 추가: https://e2b.dev/docs/code-interpreting/supported-languages
    """샌드박스에서 코드 스니펫을 평가하는 보상 함수를 반환합니다."""
    evaluation_script_template = """
    import subprocess
    import json

    def evaluate_code(code, test_cases):
        passed = 0
        total = len(test_cases)
        exec_timeout = 5

        for case in test_cases:
            process = subprocess.run(
                ["python3", "-c", code],
                input=case["input"],
                text=True,
                capture_output=True,
                timeout=exec_timeout
            )

            if process.returncode != 0:  # 실행 중 오류 발생
                continue

            output = process.stdout.strip()

            # TODO: 정답과 비교하기 위한 적절한 검증기 구현. 현재는 stdout의 각 줄에 대해 정확한 문자열 일치만 확인함.
            all_correct = True
            for line1, line2 in zip(output.split('\\n'), case['output'].split('\\n')):
                all_correct = all_correct and line1.strip() == line2.strip()

            if all_correct:
                passed += 1

        success_rate = (passed / total)
        return success_rate

    code_snippet = {code}
    test_cases = json.loads({test_cases})

    evaluate_code(code_snippet, test_cases)
    """
    code_snippets = [
        extract_code(completion[-1]["content"]) for completion in completions
    ]
    verification_info = kwargs["verification_info"]
    scripts = [
        evaluation_script_template.format(
            code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"]))
        )
        for code, info in zip(code_snippets, verification_info)
    ]

    language = verification_info[0]["language"]

    if not all(v["language"] == language for v in verification_info):
        raise ValueError(
            "모든 verification_info는 동일한 언어를 가져야 합니다", verification_info
        )
    try:
        rewards = run_async_from_sync(scripts, language)

    except Exception as e:
        print(f"E2B 실행기 오류: {e}")
        rewards = [0.0] * len(completions)

    return rewards


def get_code_format_reward(language: str = "python"):
    """코드 형식 보상 함수를 생성하는 팩토리 함수입니다.

    지정된 프로그래밍 언어에 맞게 코드 답변이 올바른 형식을 갖추었는지 확인합니다.
    <think></think> 태그 안에 추론 과정이 있고, <answer></answer> 태그 안에
    ```language``` 코드 블록이 있어야 합니다.

    Args:
        language: E2B에서 지원하는 프로그래밍 언어 https://e2b.dev/docs/code-interpreting/supported-languages

    Returns:
        function: 파라미터가 적용된 코드 형식 보상 함수
    """
    pattern = (
        rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{language}.*?```.*?\n</answer>$"
    )

    def code_format_reward(completions, **kwargs):
        """코드 형식 보상 함수로, 모델의 답변이 지정된 코드 형식을 따르는지 확인합니다.

        Args:
            completions: 모델이 생성한 답변 목록

        Returns:
            float: 각 답변에 대한 코드 형식 보상 점수 목록 (0.0 또는 1.0)
        """
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [
            re.match(pattern, content, re.DOTALL | re.MULTILINE)
            for content in completion_contents
        ]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward


def run_async_from_sync(scripts: list[str], language: str) -> list[float]:
    """동기 환경에서 비동기 함수(run_async)를 실행하기 위한 래퍼 함수입니다.

    새 이벤트 루프를 생성하고 비동기 함수를 실행하여 결과를 반환합니다.

    Args:
        scripts: 실행할 스크립트 목록
        language: 프로그래밍 언어

    Returns:
        list[float]: 각 스크립트 실행 결과의 보상 점수 목록
    """
    # 새 이벤트 루프 생성 및 설정
    try:
        # 비동기 함수 실행 및 결과 가져오기
        rewards = asyncio.run(run_async(scripts, language))
    except Exception as e:
        print(f"E2B 실행기 비동기 오류: {e}")
        raise e

    return rewards


async def run_async(scripts: list[str], language: str) -> list[float]:
    """여러 스크립트를 비동기적으로 실행합니다.

    E2B 샌드박스를 생성하고 모든 스크립트를 병렬로 실행한 후 결과를 수집합니다.

    Args:
        scripts: 실행할 스크립트 목록
        language: 프로그래밍 언어

    Returns:
        list[float]: 각 스크립트 실행 결과의 보상 점수 목록
    """
    # 수동으로 샌드박스 생성, 현재 버전에서는 컨텍스트 매니저가 없음
    sbx = await AsyncSandbox.create(timeout=30, request_timeout=3)

    # 스크립트를 동시에 실행하기 위한 태스크 목록 생성
    tasks = [run_script(sbx, script, language) for script in scripts]

    # 모든 태스크가 완료될 때까지 기다리고 완료될 때마다 결과 수집
    results = await asyncio.gather(*tasks)
    rewards = list(results)  # 결과 수집

    # 모든 태스크가 완료된 후 샌드박스 종료
    await sbx.kill()

    return rewards


async def run_script(sbx: AsyncSandbox, script: str, language: str) -> float:
    """E2B 샌드박스에서 단일 스크립트를 실행합니다.

    스크립트를 실행하고 결과 텍스트를 부동소수점으로 변환하여 반환합니다.
    실행 오류가 발생하면 0.0을 반환합니다.

    Args:
        sbx: E2B 샌드박스 인스턴스
        script: 실행할 스크립트
        language: 프로그래밍 언어

    Returns:
        float: 스크립트 실행 결과의 보상 점수
    """
    execution = await sbx.run_code(script, language=language)
    try:
        return float(execution.text)
    except (TypeError, ValueError):
        return 0.0
