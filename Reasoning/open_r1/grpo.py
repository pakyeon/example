# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Apache License, Version 2.0 라이센스에 따라 사용이 허가됩니다.
# 라이센스 사본은 다음에서 얻을 수 있습니다:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 법률에 의해 요구되거나 서면으로 동의하지 않는 한, 이 소프트웨어는
# "있는 그대로" 제공되며, 명시적이든 묵시적이든 어떠한 종류의 보증이나 조건 없이 제공됩니다.
# 라이센스에 따른 특정 언어의 권한 및 제한 사항에 대한 자세한 내용은 라이센스를 참조하십시오.

import logging
import os
import sys

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config


# 로깅 설정
logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
    """
    GRPO 훈련의 메인 함수입니다.

    매개변수:
        script_args (GRPOScriptArguments): 스크립트 관련 인수
        training_args (GRPOConfig): 훈련 관련 설정 인수
        model_args (ModelConfig): 모델 관련 설정 인수
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

    # 각 프로세스에 간단한 요약 로그
    logger.warning(
        f"프로세스 순위: {training_args.local_rank}, 장치: {training_args.device}, GPU 수: {training_args.n_gpu}"
        + f" 분산 훈련: {bool(training_args.local_rank != -1)}, 16비트 훈련: {training_args.fp16}"
    )
    logger.info(f"모델 매개변수 {model_args}")
    logger.info(f"스크립트 매개변수 {script_args}")
    logger.info(f"훈련 매개변수 {training_args}")

    # 마지막 체크포인트 확인
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"체크포인트 감지, {last_checkpoint=}에서 훈련 재개.")

    # Weights & Biases 로깅 초기화 (설정된 경우)
    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # 데이터셋 로드
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    ################
    # 토크나이저 로드
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    # 보상 함수 레지스트리에서 가져오기
    reward_funcs = get_reward_funcs(script_args.reward_funcs)

    # 대화 형식으로 포맷팅하는 함수
    def make_conversation(example):
        """
        데이터셋 예제를 대화 형식으로 변환합니다.

        매개변수:
            example (dict): 데이터셋의 예제

        반환:
            dict: 대화 형식으로 변환된 예제
        """
        prompt = []

        # 시스템 프롬프트가 제공된 경우 추가
        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        # 사용자 메시지 추가
        prompt.append({"role": "user", "content": example["problem"]})
        return {"prompt": prompt}

    # 데이터셋을 대화 형식으로 변환
    dataset = dataset.map(make_conversation)

    # 불필요한 열 제거
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    # 모델 초기화 매개변수 설정
    logger.info("*** 모델 kwargs 초기화 중 ***")
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    #############################
    # GRPO 트레이너 초기화
    #############################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
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
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # 모델 저장 및 모델 카드 생성
    ##################################
    logger.info("*** 모델 저장 ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"모델이 {training_args.output_dir}에 저장되었습니다")

    # 기타 항목을 메인 프로세스에 저장
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # 빠른 추론을 위해 k,v 캐시 복원
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # 평가
    ##########
    if training_args.do_eval:
        logger.info("*** 평가 시작 ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # Hub에 푸시
    #############
    if training_args.push_to_hub:
        logger.info("Hub에 푸시 중...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    # 명령줄 인수 파싱
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
