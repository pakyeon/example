import argparse
import json
import re
from pathlib import Path
from typing import NamedTuple
from datasets import load_dataset
from transformers import (
    Gemma3ForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
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


def load_model(model_name):
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        attn_implementation="eager",
    )

    return model, processor.tokenizer


# ===== AIME 전용 프롬프트 템플릿 =====
AIME_PROMPT = """
Solve the following math problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct'

{Question}
""".strip()


# ===== 답변 추출 함수 =====
def extract_aime_answer(response):
    # 박스 형식 및 숫자 직접 매칭
    # 모든 \boxed{} 패턴을 찾아 마지막 항목 선택
    boxed_matches = re.findall(r"\\boxed{([^}]+)}", response)
    if boxed_matches:
        raw_answer = boxed_matches[-1].strip()
    else:
        patterns = [
            r"ANSWER\s*:\s*(\d+)",  # 간단한 숫자 형식
            r"final answer is:\s*(\d+)",  # 대체 표현
        ]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                raw_answer = match.group(1).strip()
                break
        else:
            return None

    # 숫자를 3자리 문자열로 포맷팅
    try:
        num = int(raw_answer)
        return f"{num:03d}"  # 3자리, 앞에 0 채움
    except ValueError:
        return None


# ===== 평가 함수 =====
def evaluate_aime(model, tokenizer, dataset):
    correct = 0
    total = 0
    responses = []  # 생성된 응답을 저장할 리스트 추가

    # 진행률 바 초기화
    progress_bar = tqdm(
        total=len(dataset), desc="Evaluating", unit="sample", dynamic_ncols=True
    )

    model.eval()
    with torch.inference_mode():
        for idx, item in enumerate(dataset):
            # 프롬프트 생성
            prompt = AIME_PROMPT.format(Question=item["problem"])
            # 토크나이징 (attention_mask 포함)
            tokenized = tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=4096,
            ).to(model.device)

            # 답변 생성 (attention_mask 전달)
            outputs = model.generate(
                input_ids=tokenized.input_ids,
                attention_mask=tokenized.attention_mask,
                max_new_tokens=32768,
                pad_token_id=tokenizer.eos_token_id,
                temperature=1,
                top_p=0.95,
                top_k=64,
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 정답 비교
            pred = extract_aime_answer(response)
            gold_answer = str(item["answer"])

            # 응답 정보 저장
            responses.append(
                {
                    "problem": item["problem"],
                    "gold_answer": gold_answer,
                    "model_response": response,
                    "predicted_answer": pred,
                    "is_correct": pred == gold_answer,
                }
            )

            if pred and pred == gold_answer:
                correct += 1
            total += 1

            # 진행 상황 출력
            if total % 10 == 0:
                print(f"Processed: {total} | Current Accuracy: {correct/total:.2f}")

            # 진행률 업데이트
            progress_bar.update(1)
            progress_bar.set_postfix(
                {
                    "Accuracy": f"{correct/(idx+1):.2f}",  # idx는 0부터 시작
                    "Correct": correct,  # 맞춘 횟수
                    "Total": idx + 1,  # 전체 시도 횟수
                }
            )

    accuracy = correct / total

    return accuracy, responses


# ===== 메인 실행 =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./aime_results")
    args = parser.parse_args()

    # 모델 로드
    model, tokenizer = load_model(args.model)

    # 데이터셋 로드
    # dataset = load_dataset("HuggingFaceH4/aime_2024", split="train").select(range(1))
    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")

    # 현재 시간 기반 파일 이름 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_filename = f"results_{timestamp}.json"

    # 평가 수행
    accuracy, responses = evaluate_aime(model, tokenizer, dataset)

    # 결과 저장
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    result = {"aime24_accuracy": accuracy, "responses": responses}

    with open(Path(args.output_dir) / output_filename, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nFinal AIME24 Accuracy: {accuracy:.4f}")
