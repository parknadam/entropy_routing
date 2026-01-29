# KD-LoRA 성능 최적화 가이드

## 📊 문제 분석

### 실험 결과 비교
```
실험 1 (lr=2e-4, r=8, α=16, epoch=1):
  [A] LoRA ppl=18.876 ✅ 좋음

실험 2 (lr=1e-4, r=16, α=32, epoch=2):
  [A] LoRA ppl=19.089 ❌ 더 나쁨
```

### 성능 저하 원인
1. **Learning Rate 너무 낮음**: 1e-4는 KD-LoRA에 부족
2. **Rank 증가 → 더 많은 학습 필요**: Epoch 2로는 부족
3. **Batch Size 변경 효과**: 1×32가 4×8보다 안정적일 수 있음
4. **KD alpha 0.1**: Teacher 영향력이 약함

---

## 🎯 권장 실험 순서

### 1️⃣ 기준선 재현 (가장 먼저!)
```bash
python -m prune_lora.optimized_kd_lora \
  --base_dir ./7b_results/pruning/A \
  --bundles_dir ./7b_results/pruning/bundles \
  --stage 1 \
  --out_adapters ./exp1_baseline \
  --qa_dataset squad \
  --max_samples 20000 \
  --max_eval_samples 8000 \
  --seq_len 1024 \
  --lr 2e-4 \
  --epochs 1 \
  --bs 1 \
  --grad_acc 32 \
  --lora_r 8 \
  --lora_alpha 16 \
  --use_kd \
  --teacher_model meta-llama/Llama-2-7b-chat-hf \
  --teacher_4bit \
  --teacher_device cuda:1 \
  --kd_alpha 0.1 \
  --kd_T 2.0
```
**목표**: PPL < 19.0

---

### 2️⃣ Learning Rate 증가 (즉시 개선 가능)
```bash
python -m prune_lora.optimized_kd_lora \
  --base_dir ./7b_results/pruning/A \
  --bundles_dir ./7b_results/pruning/bundles \
  --stage 1 \
  --out_adapters ./exp2_higher_lr \
  --qa_dataset squad \
  --max_samples 20000 \
  --max_eval_samples 8000 \
  --seq_len 1024 \
  --lr 3e-4 \
  --epochs 1 \
  --bs 1 \
  --grad_acc 32 \
  --lora_r 8 \
  --lora_alpha 16 \
  --use_kd \
  --teacher_model meta-llama/Llama-2-7b-chat-hf \
  --teacher_4bit \
  --teacher_device cuda:1 \
  --kd_alpha 0.1 \
  --kd_T 2.0
```
**예상**: PPL 17~18 (개선!)

---

### 3️⃣ Epoch 증가 (더 충분한 학습)
```bash
python -m prune_lora.optimized_kd_lora \
  --base_dir ./7b_results/pruning/A \
  --bundles_dir ./7b_results/pruning/bundles \
  --stage 1 \
  --out_adapters ./exp3_more_epochs \
  --qa_dataset squad \
  --max_samples 20000 \
  --max_eval_samples 8000 \
  --seq_len 1024 \
  --lr 2e-4 \
  --epochs 3 \
  --bs 1 \
  --grad_acc 32 \
  --lora_r 8 \
  --lora_alpha 16 \
  --use_kd \
  --teacher_model meta-llama/Llama-2-7b-chat-hf \
  --teacher_4bit \
  --teacher_device cuda:1 \
  --kd_alpha 0.1 \
  --kd_T 2.0
```
**예상**: PPL 16~17 (큰 개선!)

---

### 4️⃣ KD Alpha 증가 (Teacher 영향력 강화)
```bash
python -m prune_lora.optimized_kd_lora \
  --base_dir ./7b_results/pruning/A \
  --bundles_dir ./7b_results/pruning/bundles \
  --stage 1 \
  --out_adapters ./exp4_higher_alpha \
  --qa_dataset squad \
  --max_samples 20000 \
  --max_eval_samples 8000 \
  --seq_len 1024 \
  --lr 2e-4 \
  --epochs 1 \
  --bs 1 \
  --grad_acc 32 \
  --lora_r 8 \
  --lora_alpha 16 \
  --use_kd \
  --teacher_model meta-llama/Llama-2-7b-chat-hf \
  --teacher_4bit \
  --teacher_device cuda:1 \
  --kd_alpha 0.3 \
  --kd_T 2.0
```
**예상**: PPL 17~18 (KD 효과 증가)

---

### 5️⃣ 최적 조합 (강력 추천! 🌟)
```bash
python -m prune_lora.optimized_kd_lora \
  --base_dir ./7b_results/pruning/A \
  --bundles_dir ./7b_results/pruning/bundles \
  --stage 1 \
  --out_adapters ./exp5_optimal \
  --qa_dataset squad \
  --max_samples 20000 \
  --max_eval_samples 8000 \
  --seq_len 1024 \
  --lr 3e-4 \
  --epochs 2 \
  --bs 1 \
  --grad_acc 32 \
  --lora_r 8 \
  --lora_alpha 16 \
  --use_kd \
  --teacher_model meta-llama/Llama-2-7b-chat-hf \
  --teacher_4bit \
  --teacher_device cuda:1 \
  --kd_alpha 0.2 \
  --kd_T 2.0
```
**예상**: PPL 15~16 (최고 성능!)

---

### 6️⃣ Rank 16 (충분한 학습 시)
```bash
python -m prune_lora.optimized_kd_lora \
  --base_dir ./7b_results/pruning/A \
  --bundles_dir ./7b_results/pruning/bundles \
  --stage 1 \
  --out_adapters ./exp6_rank16 \
  --qa_dataset squad \
  --max_samples 20000 \
  --max_eval_samples 8000 \
  --seq_len 1024 \
  --lr 2e-4 \
  --epochs 4 \
  --bs 1 \
  --grad_acc 32 \
  --lora_r 16 \
  --lora_alpha 32 \
  --use_kd \
  --teacher_model meta-llama/Llama-2-7b-chat-hf \
  --teacher_4bit \
  --teacher_device cuda:1 \
  --kd_alpha 0.2 \
  --kd_T 2.0
```
**예상**: PPL 15~17 (Epoch 4 필요!)

---

### 7️⃣ Temperature 조정 (선택적)
```bash
python -m prune_lora.optimized_kd_lora \
  --base_dir ./7b_results/pruning/A \
  --bundles_dir ./7b_results/pruning/bundles \
  --stage 1 \
  --out_adapters ./exp7_temp3 \
  --qa_dataset squad \
  --max_samples 20000 \
  --max_eval_samples 8000 \
  --seq_len 1024 \
  --lr 2e-4 \
  --epochs 1 \
  --bs 1 \
  --grad_acc 32 \
  --lora_r 8 \
  --lora_alpha 16 \
  --use_kd \
  --teacher_model meta-llama/Llama-2-7b-chat-hf \
  --teacher_4bit \
  --teacher_device cuda:1 \
  --kd_alpha 0.1 \
  --kd_T 3.0
```
**효과**: Softer targets (미세 조정용)

---

## 🔥 빠른 개선을 위한 TOP 3 추천

### 🥇 1위: 실험 5 (최적 조합)
- **LR 증가** + **Epoch 증가** + **KD alpha 증가**
- 가장 빠르고 확실한 개선 예상

### 🥈 2위: 실험 3 (더 많은 Epoch)
- 단순하지만 효과적
- Epoch 3만으로도 큰 개선

### 🥉 3위: 실험 2 (LR 증가)
- 가장 빠른 실험 (Epoch 1)
- 즉시 개선 확인 가능

---

## 📈 하이퍼파라미터 영향도

| 파라미터 | 현재값 | 권장값 | 영향도 |
|---------|--------|--------|--------|
| **Learning Rate** | 1e-4 | 2e-4 ~ 3e-4 | 🔥🔥🔥 높음 |
| **Epochs** | 2 | 2~4 | 🔥🔥🔥 높음 |
| **KD Alpha** | 0.1 | 0.2~0.3 | 🔥🔥 중간 |
| **LoRA Rank** | 16 | 8 (효율) or 16 (성능) | 🔥🔥 중간 |
| **Temperature** | 2.0 | 2.0~3.0 | 🔥 낮음 |
| **Batch Size** | 4×8 | 1×32 | 🔥 낮음 |

---

## 💡 핵심 인사이트

1. **Rank ≠ 무조건 좋음**: Rank 8이 16보다 효율적일 수 있음
2. **LR은 충분히 높여야**: KD-LoRA는 2e-4 ~ 3e-4 필요
3. **Rank 증가 시 Epoch 증가 필수**: 2배 rank → 2배 epoch
4. **KD Alpha는 0.2~0.3 권장**: Teacher 지식 활용 증가
5. **Batch Size는 1×32 유지**: 더 안정적인 gradient

---

## 🎓 이론적 배경

### KD-LoRA Loss
```
Loss = α × KD_soft + (1-α) × CE_hard

α = 0.1 → Teacher 10%, Hard labels 90%
α = 0.3 → Teacher 30%, Hard labels 70%
```

### Rank와 파라미터 수
```
Rank 8:  ~2.9M params
Rank 16: ~5.8M params (2배)
→ 학습 시간도 2배 필요!
```

### Learning Rate 선택
```
LoRA: 1e-4 ~ 5e-4 (일반적)
KD-LoRA: 2e-4 ~ 3e-4 (soft + hard 동시 학습)
```

---

## 🚀 빠른 시작 (복붙용)

**가장 추천하는 명령어:**
```bash
python -m prune_lora.optimized_kd_lora \
  --base_dir ./7b_results/pruning/A \
  --bundles_dir ./7b_results/pruning/bundles \
  --stage 1 \
  --out_adapters ./best_result \
  --qa_dataset squad \
  --max_samples 20000 \
  --max_eval_samples 8000 \
  --seq_len 1024 \
  --lr 3e-4 \
  --epochs 2 \
  --bs 1 \
  --grad_acc 32 \
  --lora_r 8 \
  --lora_alpha 16 \
  --use_kd \
  --teacher_model meta-llama/Llama-2-7b-chat-hf \
  --teacher_4bit \
  --teacher_device cuda:1 \
  --kd_alpha 0.2 \
  --kd_T 2.0
```

**예상 PPL: 15~16 (기존 18.9에서 크게 개선!)**