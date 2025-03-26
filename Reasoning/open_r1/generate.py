# Copyright 2025 HuggingFace 팀. 모든 권리 보유.
#
# Apache License 2.0 (이하 "라이선스")에 따라 라이선스가 부여됩니다.
# 라이선스를 준수하지 않는 한 이 파일을 사용할 수 없습니다.
# 라이선스 사본은 다음에서 얻을 수 있습니다:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 관련 법률에서 요구하거나 서면으로 동의하지 않는 한, 이 라이선스에 따라 배포되는 소프트웨어는
# 명시적이든 묵시적이든 어떠한 종류의 보증이나 조건 없이 "있는 그대로" 배포됩니다.
# 라이선스에 따른 특정 언어의 권한 및 제한사항에 대해서는 라이선스를 참조하십시오.

from typing import Optional

from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import StepResources
from distilabel.steps.tasks import TextGeneration


# 이 함수는 텍스트 생성을 위한 distilabel 파이프라인을 구축합니다.
# 주요 목적은 지정된 모델과 설정을 사용하여 텍스트 생성 파이프라인을 생성하는 것입니다.
def build_distilabel_pipeline(
    model: str,  # 사용할 모델 이름
    base_url: str = "http://localhost:8000/v1",  # vLLM 서버 URL
    prompt_column: Optional[str] = None,  # 입력 데이터셋에서 프롬프트로 사용할 열 이름
    prompt_template: str = "{{ instruction }}",  # 프롬프트 형식을 지정하는 템플릿
    temperature: Optional[
        float
    ] = None,  # 생성 다양성 제어 파라미터 (높을수록 더 무작위적)
    top_p: Optional[float] = None,  # 토큰 샘플링 임계값 (누적 확률)
    max_new_tokens: int = 8192,  # 생성할 최대 토큰 수
    num_generations: int = 1,  # 각 입력에 대한 생성 횟수
    input_batch_size: int = 64,  # 입력 처리 배치 크기
    client_replicas: int = 1,  # 병렬 처리를 위한 클라이언트 복제본 수
    timeout: int = 900,  # 요청 타임아웃 (초)
    retries: int = 0,  # 실패한 요청에 대한 재시도 횟수
) -> Pipeline:
    # 생성 매개변수를 담을 딕셔너리 초기화
    generation_kwargs = {"max_new_tokens": max_new_tokens}

    # temperature가 제공된 경우 생성 매개변수에 추가
    if temperature is not None:
        generation_kwargs["temperature"] = temperature

    # top_p가 제공된 경우 생성 매개변수에 추가
    if top_p is not None:
        generation_kwargs["top_p"] = top_p

    # Ray를 사용하는 파이프라인 생성 - Ray는 분산 컴퓨팅 프레임워크
    with Pipeline().ray() as pipeline:
        # 텍스트 생성 단계 추가
        TextGeneration(
            # OpenAI 호환 API를 사용하는 언어 모델 인스턴스 생성
            llm=OpenAILLM(
                base_url=base_url,  # vLLM 서버 URL
                api_key="something",  # 더미 API 키 (vLLM 서버는 실제로 검증하지 않음)
                model=model,  # 사용할 모델 이름
                timeout=timeout,  # 요청 타임아웃
                max_retries=retries,  # 최대 재시도 횟수
                generation_kwargs=generation_kwargs,  # 생성 설정
            ),
            template=prompt_template,  # 프롬프트 템플릿
            # 입력 데이터셋 열과 템플릿 변수 간의 매핑
            input_mappings=(
                {"instruction": prompt_column} if prompt_column is not None else {}
            ),
            input_batch_size=input_batch_size,  # 입력 처리 배치 크기
            num_generations=num_generations,  # 각 입력에 대한 생성 횟수
            group_generations=True,  # 다중 생성 결과를 그룹화
            resources=StepResources(replicas=client_replicas),  # 병렬 처리 리소스 설정
        )

    return pipeline


# 스크립트가 직접 실행될 때 실행되는 메인 블록
if __name__ == "__main__":
    import argparse

    from datasets import load_dataset

    # 명령행 인수 파서 설정
    parser = argparse.ArgumentParser(
        description="DeepSeek R1을 사용하여 응답을 생성하기 위한 distilabel 파이프라인 실행"
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        required=True,
        help="로드할 HuggingFace 데이터셋",
    )
    parser.add_argument(
        "--hf-dataset-config",
        type=str,
        required=False,
        help="사용할 데이터셋 구성",
    )
    parser.add_argument(
        "--hf-dataset-split",
        type=str,
        default="train",
        help="사용할 데이터셋 분할",
    )
    parser.add_argument(
        "--prompt-column",
        type=str,
        default="prompt",
        help="프롬프트로 사용할 데이터셋 열",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="{{ instruction }}",
        help="프롬프트 형식을 지정하는 템플릿 문자열",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="생성에 사용할 모델 이름",
    )
    parser.add_argument(
        "--vllm-server-url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM 서버의 URL",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="생성 다양성 제어",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="생성을 위한 top-p 값 (토큰 샘플링 임계값)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="생성할 최대 새 토큰 수",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=1,
        help="문제당 생성 횟수",
    )
    parser.add_argument(
        "--input-batch-size",
        type=int,
        default=64,
        help="입력 처리 배치 크기",
    )
    parser.add_argument(
        "--client-replicas",
        type=int,
        default=1,
        help="병렬 처리를 위한 클라이언트 복제본 수",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="요청 타임아웃(초) (기본값: 600)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="실패한 요청에 대한 재시도 횟수 (기본값: 0)",
    )
    parser.add_argument(
        "--hf-output-dataset",
        type=str,
        required=False,
        help="결과를 업로드할 HuggingFace 저장소",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="HF Hub에 푸시할 때 출력 데이터셋을 비공개로 설정할지 여부",
    )

    # 명령행 인수 파싱
    args = parser.parse_args()

    # 실행 인수 출력
    print("\n다음 인수로 실행 중:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    # 지정된 HuggingFace 데이터셋 로드
    print(
        f"'{args.hf_dataset}' (config: {args.hf_dataset_config}, split: {args.hf_dataset_split}) 데이터셋 로드 중..."
    )
    dataset = load_dataset(
        args.hf_dataset, args.hf_dataset_config, split=args.hf_dataset_split
    )
    print("데이터셋 로드 완료!")

    # 설정한 파라미터로 distilabel 파이프라인 구축
    pipeline = build_distilabel_pipeline(
        model=args.model,
        base_url=args.vllm_server_url,
        prompt_template=args.prompt_template,
        prompt_column=args.prompt_column,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        num_generations=args.num_generations,
        input_batch_size=args.input_batch_size,
        client_replicas=args.client_replicas,
        timeout=args.timeout,
        retries=args.retries,
    )

    # 생성 파이프라인 실행
    print("생성 파이프라인 실행 중...")
    distiset = pipeline.run(
        dataset=dataset,
        dataset_batch_size=args.input_batch_size * 1000,  # 데이터셋 처리 배치 크기
        use_cache=False,  # 캐시 사용 안 함
    )
    print("생성 파이프라인 완료!")

    # 결과 데이터셋을 HuggingFace Hub에 업로드 (지정된 경우)
    if args.hf_output_dataset:
        print(f"결과 데이터셋을 '{args.hf_output_dataset}'에 업로드 중...")
        distiset.push_to_hub(args.hf_output_dataset, private=args.private)
        print("데이터셋 업로드 완료!")
