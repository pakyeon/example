import argparse
import json
import re
from pathlib import Path
from typing import NamedTuple
from datasets import load_dataset
from transformers import BitsAndBytesConfig
from transformers import TextStreamer
from peft import PeftModel

import torch
from tqdm import tqdm
from datetime import datetime


# ===== 데이터 구조 정의 =====
class Doc(NamedTuple):
    query: str
    choices: list
    gold_index: int


# ===== 모델 및 토크나이저 로드 =====
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


# unsloth 모델을 불러올 때 사용
def load_unsloth_model(model_name):
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    return model, tokenizer


# Gemma3 멀티모달 모델을 불러올 때 사용
def load_multimodal_model(model_name):
    from transformers import (
        Gemma3ForConditionalGeneration,
        AutoProcessor,
    )

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        attn_implementation="eager",
    )

    return model, processor.tokenizer


def load_merge_model(model_name, adapter_path):
    from transformers import (
        Gemma3ForConditionalGeneration,
        AutoProcessor,
    )

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    base_model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # 이 옵션 추가
    )

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
    )

    model = model.merge_and_unload()

    return model, processor.tokenizer


# ===== MATH 전용 프롬프트 템플릿 =====
MATH_PROMPT = """
Solve the following math problem, given in LaTeX format, step by step. Ensure your solution is efficient, clear, and logically structured. Explicitly show all steps and reasoning.

The last line of your response must strictly follow this format:
'Therefore, the final answer is: $\\boxed{{ANSWER}}$.'

where ANSWER is the final number or expression that solves the problem. 

Here is the problem:
{Question}
""".strip()


# ===== 답변 추출 함수 =====
def extract_answer(text):
    boxed_index = text.rfind("\\boxed{")
    if boxed_index == -1:
        return ""

    start_idx = boxed_index + 7
    brace_count = 1
    end_idx = start_idx

    while brace_count > 0 and end_idx < len(text):
        if text[end_idx] == "{":
            brace_count += 1
        elif text[end_idx] == "}":
            brace_count -= 1
        end_idx += 1

    answer = text[start_idx : end_idx - 1].strip()
    return re.sub(r"\s+", "", answer)  # 내부 공백 제거


# ===== 평가 함수 =====
def evaluate(model, tokenizer, dataset, batch_size=16):
    correct = 0
    total = 0
    responses = []

    # 배치 생성
    dataset = dataset.map(
        lambda _, idx: {"batch_id": idx // batch_size}, with_indices=True
    )
    batched_dataset = dataset.to_pandas().groupby("batch_id")

    progress_bar = tqdm(
        total=len(dataset), desc="Evaluating", unit="sample", dynamic_ncols=True
    )

    model.eval()
    with torch.inference_mode():
        for batch_id, batch_group in batched_dataset:
            batch = dataset.select(batch_group.index.tolist())

            # 배치 프롬프트 생성
            chats = [
                [
                    {
                        "role": "user",
                        "content": MATH_PROMPT.format(Question=item["problem"]),
                    }
                ]
                for item in batch
            ]
            prompts = [
                tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
                for chat in chats
            ]

            # 배치 토크나이징
            tokenized = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(model.device)

            # 배치 생성
            outputs = model.generate(
                **tokenized,
                max_new_tokens=4096,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                eos_token_id=[1, 106],
            )

            # 배치 결과 처리
            for i, output in enumerate(outputs):
                response = tokenizer.decode(output, skip_special_tokens=True)
                pred = extract_answer(response)
                gold_answer = str(batch[i]["answer"])

                # 공백 제거 비교
                pred_clean = re.sub(r"\s+", "", pred)
                gold_clean = re.sub(r"\s+", "", gold_answer)

                # 응답 기록
                responses.append(
                    {
                        "problem": batch[i]["problem"],
                        "gold_answer": gold_answer,
                        "model_response": response,
                        "predicted_answer": pred,
                        "is_correct": pred_clean == gold_clean,
                    }
                )

                if pred_clean == gold_clean:
                    correct += 1
                total += 1

                progress_bar.update(1)
                progress_bar.set_postfix(
                    {
                        "Accuracy": f"{correct/total:.2f}",
                        "Correct": correct,
                        "Total": total,
                    }
                )

    accuracy = correct / total
    return accuracy, responses


# ===== 메인 실행 =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./aime_results")
    args = parser.parse_args()

    # 모델 로드
    model, tokenizer = load_multimodal_model(args.model)
    # model, tokenizer = load_unsloth_model(args.model)
    # model, tokenizer = load_merge_model(args.model, args.adapter)

    # 데이터셋 로드
    # dataset = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(1))
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    # 현재 시간 기반 파일 이름 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_filename = f"results_{timestamp}.json"

    # 평가 수행
    accuracy, responses = evaluate(model, tokenizer, dataset)

    # 결과 저장
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    result = {"accuracy": accuracy, "responses": responses}

    with open(Path(args.output_dir) / output_filename, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nFinal Accuracy: {accuracy:.4f}")
