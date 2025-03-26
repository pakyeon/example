# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# 아파치 라이선스 2.0 버전(이하 "라이선스")에 따라 라이선스가 부여됩니다.
# 라이선스를 준수하지 않는 한 이 파일을 사용할 수 없습니다.
# 라이선스 사본은 다음에서 얻을 수 있습니다:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 관련 법률에서 요구하거나 서면으로 동의하지 않는 한, 라이선스에 따라 배포되는 소프트웨어는
# "있는 그대로" 배포되며, 명시적이든 묵시적이든 어떠한 종류의 보증이나 조건 없이 제공됩니다.
# 라이선스에 따른 특정 언어의 권한 및 제한사항에 대한 자세한 내용은 라이선스를 참조하십시오.

"""
디코더 언어 모델을 위한 지도 학습 미세 조정 스크립트.

사용법:

# 8개의 H100 GPU가 있는 1개 노드에서 실행
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill
"""

# 필요한 라이브러리 임포트
import logging
import os
import sys

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import SFTConfig
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


# 로거 설정
logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
    """
    메인 함수: SFT(Supervised Fine-Tuning) 프로세스를 실행합니다.

    Args:
        script_args: 스크립트 관련 인자(데이터셋, 분할 등)
        training_args: 훈련 관련 인자(배치 크기, 에폭 등)
        model_args: 모델 관련 인자(모델 이름, PEFT 설정 등)
    """
    # 재현성을 위한 시드 설정
    set_seed(training_args.seed)

    ###############
    # 로깅 설정
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 인자 정보 로깅
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # 마지막 체크포인트 확인
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(
            f"체크포인트가 감지되었습니다. {last_checkpoint=}에서 훈련을 재개합니다."
        )

    # Weights & Biases 로깅 초기화 (설정된 경우)
    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ################
    # 데이터셋 로드
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    ################
    # 토크나이저 로드
    ################
    tokenizer = get_tokenizer(model_args, training_args)
    tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰을 EOS 토큰으로 설정

    ###################
    # 모델 초기화 매개변수
    ###################
    logger.info("*** 모델 매개변수 초기화 중 ***")
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=(
            False if training_args.gradient_checkpointing else True
        ),  # 그래디언트 체크포인팅 사용 시 캐시 비활성화
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    ############################
    # SFT 트레이너 초기화
    ############################
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        processing_class=tokenizer,
        peft_config=get_peft_config(
            model_args
        ),  # PEFT(Parameter-Efficient Fine-Tuning) 설정
        callbacks=get_callbacks(training_args, model_args),  # 콜백 함수 설정
    )

    ###############
    # 훈련 루프
    ###############
    logger.info("*** 훈련 시작 ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)  # 훈련 메트릭 로깅
    trainer.save_metrics("train", metrics)  # 훈련 메트릭 저장
    trainer.save_state()  # 훈련 상태 저장

    ##################################
    # 모델 저장 및 모델 카드 생성
    ##################################
    logger.info("*** 모델 저장 ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"모델이 {training_args.output_dir}에 저장되었습니다")

    # 메인 프로세스에서 기타 모든 것 저장
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)  # 모델 카드 생성
        # 빠른 추론을 위해 키-값 캐시 복원
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # 평가
    ##########
    if training_args.do_eval:
        logger.info("*** 평가 ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)  # 평가 메트릭 로깅
        trainer.save_metrics("eval", metrics)  # 평가 메트릭 저장

    #############
    # hub에 푸시
    #############
    if training_args.push_to_hub:
        logger.info("hub에 푸시 중...")
        trainer.push_to_hub(**kwargs)  # 모델을 Hugging Face Hub에 업로드


if __name__ == "__main__":
    """
    스크립트의 진입점:
    인자를 파싱하고 main 함수를 호출합니다.
    """
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))  # 인자 파서 초기화
    script_args, training_args, model_args = parser.parse_args_and_config()  # 인자 파싱
    main(script_args, training_args, model_args)  # 메인 함수 호출
