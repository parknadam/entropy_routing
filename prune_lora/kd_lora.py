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

# vast ai 훈련용
python -m prune_lora.kd_lora \
  --base_dir /dev/shm/7b_results/pruning/A \
  --stage 1 \
  --out_adapters /dev/shm/kd_lora_results/adapters \
  --qa_dataset squad \
  --max_samples 20000 \
  --max_eval_samples 8000 \
  --lr 1e-5 \
  --seq_len 1024 \
  --epochs 1 \
  --bs 1 \
  --grad_acc 32

"""


import os
import argparse
import torch
import math
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
        ctx = ex.get("context", "")
        q = ex.get("question", "")
        ans = ex.get("answers", {}).get("text", [""])[0]

        reserve_ans = 64  # 답변을 위해 최소 64토큰 공간 확보(조절 가능)

        prompt = f"Context: {ctx}\nQuestion: {q}\nAnswer:"

        prompt_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=max(1, args.seq_len - reserve_ans),
            padding=False,
        )["input_ids"]
        prompt_trunc = tokenizer.decode(prompt_ids, skip_special_tokens=True)

        full_text = prompt + " " + ans + tokenizer.eos_token

        inputs = tokenizer(
            full_text,
            truncation=True,
            max_length=args.seq_len,
            padding="max_length",
        )

        # 1) labels 생성 (tokenizer는 labels를 안 줌)
        labels = inputs["input_ids"].copy()

        # 프롬프트 부분 마스킹
        prompt_ids2 = tokenizer(prompt_trunc, truncation=True, max_length=args.seq_len, padding=False)["input_ids"]
        prompt_len = min(len(prompt_ids2), len(labels))
        for i in range(prompt_len):
            labels[i] = -100

        # pad 마스킹
        attn = inputs["attention_mask"]
        for i in range(len(labels)):
            if attn[i] == 0:
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
    def __init__(self, *args, teacher_model=None, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model.eval()
        self.alpha = alpha
        self.T = temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]

        # student forward
        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        s_logits_full = student_outputs.logits  # [B, L, V]

        # ✅ CausalLM shift 먼저
        s_logits = s_logits_full[:, :-1, :].contiguous()
        labels_s = labels[:, 1:].contiguous()
        attn_s   = inputs["attention_mask"][:, 1:].contiguous()

        # ✅ supervised token mask
        mask = (labels_s != -100) & (attn_s == 1)
        tok_n = int(mask.sum().item())

        # ✅ tok_n=0 방어 (teacher도 안 돌림)
        if tok_n == 0:
            log_steps = getattr(self.args, "logging_steps", 0)
            gs = getattr(self.state, "global_step", 0)
            if log_steps and log_steps > 0 and (gs % log_steps == 0):
                self.log({"skip_batch_no_supervised_tokens": 1.0, "tok_n": 0.0})
            loss = s_logits_full.sum() * 0.0  # grad-safe
            return (loss, student_outputs) if return_outputs else loss

        # teacher forward (필요할 때만)
        with torch.no_grad():
            t_inputs = {
                "input_ids": inputs["input_ids"].to(self.teacher_model.device),
                "attention_mask": inputs["attention_mask"].to(self.teacher_model.device),
            }
            t_logits_full = self.teacher_model(**t_inputs).logits.to(s_logits_full.device)

        t_logits = t_logits_full[:, :-1, :].contiguous()

        # gather
        s = s_logits[mask]      # [N, V]
        t = t_logits[mask]      # [N, V]
        y = labels_s[mask]      # [N]

        # fp32 for stability (✅ 먼저 float로 만들기)
        s_fp32 = s.float()
        t_fp32 = t.float()

        # (선택) logits 범위 제한: 안정성 크게 올라감
        s_fp32 = s_fp32.clamp(-50, 50)
        t_fp32 = t_fp32.clamp(-50, 50)

        # fp32 for stability
        T = self.T

        log_s = F.log_softmax(s_fp32 / T, dim=-1)
        log_t = F.log_softmax(t_fp32 / T, dim=-1)

        soft_loss = F.kl_div(
            log_s,
            log_t,
            reduction="batchmean",
            log_target=True,   # ✅ 중요
        ) * (T * T)

        hard_loss = F.cross_entropy(s_fp32, y)
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        # logging
        log_steps = getattr(self.args, "logging_steps", 0)
        gs = getattr(self.state, "global_step", 0)
        if log_steps and log_steps > 0 and (gs % log_steps == 0):
            hard = hard_loss.detach().float().item()
            soft = soft_loss.detach().float().item()
            self.log({
                "tok_n": float(tok_n),
                "kd_soft": soft,
                "kd_hard": hard,
                "kd_ppl": math.exp(hard),
            })

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
    # lora_config = LoraConfig(
    #     r=args.rank,
    #     target_modules=["q_proj", "v_proj"],
    #     task_type=TaskType.CAUSAL_LM,
    # )
    lora_config = LoraConfig(
        r=8,                    # stageA랑 동일
        lora_alpha=16,          # 보통 2*r
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj"
        ],
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
        bf16=True,
        fp16=False,

        # 로그
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,

        # 중간 평가(잘 되고 있는지 확인용)
        eval_strategy="steps",
        eval_steps=200,

        # 중간 저장(혹시 망하면 여기서 되돌릴 수 있게)
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,

        # 안정성/가독성
        max_grad_norm=1.0,
        remove_unused_columns=False,  # inputs에 labels/attention_mask 유지
        report_to=["tensorboard"],    # 싫으면 [] 로
        logging_dir=os.path.join(args.out_adapters, "tb"),
    )

    trainer = KDTrainer(
        model=student,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
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