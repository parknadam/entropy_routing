"""
python -m prune_lora.kd_lora \
  --base_dir ./7b_results/pruning/A \
  --stage 1 \
  --out_adapters ./kd_lora_results/adapters \
  --qa_dataset squad \
  --max_samples 20000 \
  --max_eval_samples 8000 \
  --seq_len 1024 \
  --epochs 1 \
  --bs 1 \
  --grad_acc 32

python -m prune_lora.kd_lora \
  --base_dir /dev/shm/7b_results/pruning/A \
  --stage 1 \
  --out_adapters /dev/shm/kd_lora_results/adapters \
  --qa_dataset squad \
  --max_samples 20000 \
  --max_eval_samples 8000 \
  --seq_len 1024 \
  --epochs 1 \
  --bs 1 \
  --grad_acc 32

"""


import os
import argparse
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import LoraConfig, TaskType, PeftModel, get_peft_model
import inspect

# -----------------------------
# 1) Argument Parser (요청하신 커맨드라인 대응)
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("Progressive KD-LoRA for SQuAD")
    
    # 경로 및 스테이지 관련
    p.add_argument("--base_dir", type=str, required=True, help="Base/Student model path")
    p.add_argument("--bundles_dir", type=str, default=None, help="Pruning bundles directory")
    p.add_argument("--stage", type=str, default="1", help="Current training stage (e.g., 1, 2, A, B)")
    p.add_argument("--out_adapters", type=str, required=True, help="Output directory for adapters")
    
    # 데이터셋 관련
    p.add_argument("--qa_dataset", type=str, default="squad", help="Dataset name (e.g. squad)")
    p.add_argument("--max_samples", type=int, default=20000)
    p.add_argument("--max_eval_samples", type=int, default=8000)
    p.add_argument("--seq_len", type=int, default=1024)
    
    # 학습 하이퍼파라미터 (단축어 대응)
    p.add_argument("--epochs", type=int, default=1, dest="num_train_epochs")
    p.add_argument("--bs", type=int, default=1, dest="train_batch_size")
    p.add_argument("--grad_acc", type=int, default=32, dest="gradient_accumulation_steps")
    p.add_argument("--lr", type=float, default=2e-4, dest="learning_rate")
    
    # KD & LoRA 설정
    p.add_argument("--teacher_model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.5, help="KD alpha weight")
    p.add_argument("--fp16", action="store_true", default=True)

    return p.parse_args()

# -----------------------------
# 2) SQuAD 데이터셋 처리 로직
# -----------------------------
def load_qa_dataset(tokenizer, args):
    print(f"[Info] Loading dataset: {args.qa_dataset}")
    ds = load_dataset(args.qa_dataset, split="train")
    eval_ds = load_dataset(args.qa_dataset, split="validation")

    if args.max_samples:
        ds = ds.shuffle(seed=42).select(range(min(args.max_samples, len(ds))))
    if args.max_eval_samples:
        eval_ds = eval_ds.shuffle(seed=42).select(range(min(args.max_eval_samples, len(eval_ds))))

    def preprocess(ex):
        # SQuAD 특화 포맷팅
        ctx = ex.get("context", "")
        q = ex.get("question", "")
        ans = ex.get("answers", {}).get("text", [""])[0]
        
        prompt = f"Context: {ctx}\nQuestion: {q}\nAnswer:"
        full_text = prompt + " " + ans + tokenizer.eos_token
        
        inputs = tokenizer(full_text, truncation=True, max_length=args.seq_len, padding="max_length")
        prompt_ids = tokenizer(prompt, truncation=True, max_length=args.seq_len)["input_ids"]
        
        labels = list(inputs["input_ids"])
        # 프롬프트 부분은 -100으로 마스킹 (Loss 계산 제외)
        for i in range(len(prompt_ids)):
            labels[i] = -100
            
        inputs["labels"] = labels
        return inputs

    ds = ds.map(preprocess, remove_columns=ds.column_names)
    eval_ds = eval_ds.map(preprocess, remove_columns=eval_ds.column_names)
    return ds, eval_ds

# -----------------------------
# 3) KD Trainer 정의
# -----------------------------
class KDTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model.eval()
        self.alpha = alpha

    # 이 부분의 인자를 수정합니다.
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.get("labels")
        student_outputs = model(**inputs)
        s_logits = student_outputs.logits

        with torch.no_grad():
            # Teacher 모델 연산 (GPU 1)
            t_inputs = {k: v.to(self.teacher_model.device) for k, v in inputs.items() if k != "labels"}
            t_logits = self.teacher_model(**t_inputs).logits.to(s_logits.device)

        # Distillation Loss 계산
        mask = labels.ne(-100)
        s_logits_sl = s_logits[mask]
        t_logits_sl = t_logits[mask]
        labels_sl = labels[mask]

        soft_loss = F.kl_div(
            F.log_softmax(s_logits_sl / 2.0, dim=-1),
            F.softmax(t_logits_sl / 2.0, dim=-1),
            reduction="batchmean"
        ) * (2.0 ** 2)
        
        hard_loss = F.cross_entropy(s_logits_sl, labels_sl)
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return (loss, student_outputs) if return_outputs else loss
    

# -----------------------------
# 4) Main 실행 부
# -----------------------------
def main():
    args = parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 모델 로드
    print(f"[Step] Loading Student from {args.base_dir}")
    student = AutoModelForCausalLM.from_pretrained(args.base_dir, torch_dtype=torch.float16, device_map={"": 0})
    
    print(f"[Step] Loading Teacher from {args.teacher_model}")
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, torch_dtype=torch.float16, load_in_4bit=True, device_map={"": 1})

    # LoRA 설정
    lora_config = LoraConfig(
        r=args.rank,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    student = get_peft_model(student, lora_config)
    student.print_trainable_parameters()

    # 데이터 로드
    train_ds, eval_ds = load_qa_dataset(tokenizer, args)

    # 훈련 인자 설정
    train_args = TrainingArguments(
        output_dir=args.out_adapters,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        logging_steps=10,
        eval_strategy="epoch",  # 'evaluation_strategy'를 'eval_strategy'로 변경
        save_strategy="no"
    )

    trainer = KDTrainer(
        model=student,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        teacher_model=teacher,
        alpha=args.alpha
    )

    print(f"[Start] Training Stage {args.stage}...")
    trainer.train()
    
    # 어댑터 저장
    final_save_path = os.path.join(args.out_adapters, f"stage_{args.stage}")
    student.save_pretrained(final_save_path)
    print(f"[Done] Adapter saved to {final_save_path}")

if __name__ == "__main__":
    main()