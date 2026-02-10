# Mistral Layer Pruning

Mistral 7B 모델을 위한 레이어 프루닝 구현입니다. LLaMA-2 프루닝 코드를 Mistral에 맞게 수정했습니다.

논문: ["The Unreasonable Ineffectiveness of the Deeper Layers" (ICLR 2025)](https://arxiv.org/abs/2403.17887)

## 🔑 핵심 차이점: Mistral vs LLaMA-2

### 아키텍처
- **LLaMA-2**: Standard Multi-Head Attention (MHA)
- **Mistral**: Grouped-Query Attention (GQA) + Sliding Window Attention
- **Layer 구조**: 둘 다 `model.model.layers` 사용 (동일!)

### Hyperparameters (논문 Table 1)
| Parameter | LLaMA-2 7B | Mistral 7B |
|-----------|------------|------------|
| Learning Rate | 3e-4 | **3e-6** (100배 낮음) |
| LoRA Rank | 2 | **4** (2배 높음) |
| LoRA Alpha | 2 | 4 |
| FFN Modules | gate_proj, down_proj, up_proj | 동일 |
| Num Layers | 32 | 32 |
| Hidden Size | 4096 | 4096 |

### Pruning 결과 (논문)
- **Mistral 7B**: 30-40% 레이어 제거 가능 (minimal MMLU degradation)
- **LLaMA-2 7B**: 25-35% 레이어 제거 가능
- **공통점**: 깊은 레이어일수록 유사도 높음, 마지막 레이어는 필수

---

## 📦 설치

```bash
pip install torch transformers datasets safetensors peft accelerate
```

---

## 🚀 사용법

### 1. Layer Pruning (A 모델 + B/C 번들 생성)

```bash
# Mistral 7B Pruning
python -m pruning.layeronly_drop \
  --model mistralai/Mistral-7B-v0.1 \
  --device cuda:0 \
  --drop_frac 0.25 \
  --keep_last_layer \
  --nsamples 64 \
  --seqlen 1024 \
  --max_batches 32 \
  --save_dir ./mistral_results/pruning/A \
  --save_removed_dir ./mistral_results/pruning/bundles
```

**출력 구조:**
```
mistral_results/
├── pruning/
│   ├── A/                          # 프루닝된 모델 (24 layers)
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── manifest.json           # 프루닝 정보
│   │   ├── layers_map.json         # 레이어-키 매핑
│   │   └── original_config/        # 원본 설정
│   └── bundles/
│       ├── B/                      # 제거된 레이어 그룹 1
│       │   ├── layer_021.safetensors
│       │   ├── layer_022.safetensors
│       │   └── bundle_meta.json
│       └── C/                      # 제거된 레이어 그룹 2
│           ├── layer_025.safetensors
│           ├── layer_026.safetensors
│           └── bundle_meta.json
```

### 2. Dynamic Stage Routing (A/AB/FULL)

```bash
# Entropy 기반 동적 라우팅
python -m pruning.entropy_cli \
  --base_model ./mistral_results/pruning/A \
  --bundles_dir ./mistral_results/pruning/bundles \
  --max_new_tokens 128 \
  --device cuda:0
```

**작동 방식:**
1. **Stage A**: Pruned model만 (24 layers)
2. **Stage AB**: A + Bundle B 복원 (28 layers)
3. **Stage FULL**: A + B + C 모두 복원 (32 layers)

프롬프트 복잡도에 따라 자동으로 stage 선택:
- 간단한 질문 → A (빠름)
- 중간 복잡도 → AB
- 복잡한 질문 → FULL (정확함)

---

## 🎯 모델 자동 감지

모든 스크립트는 모델 타입을 자동으로 감지합니다:

```python
# 지원 모델
- mistralai/Mistral-7B-v0.1
- mistralai/Mistral-7B-Instruct-v0.2
- meta-llama/Llama-2-7b-hf
- meta-llama/Llama-2-7b-chat-hf
- meta-llama/Llama-2-13b-hf
- facebook/opt-6.7b
```

---

## 📊 Pruning 전략

### Angular Distance 기반 선택
```python
d(ℓ) = (1/π) * arccos(mean_cos(x(ℓ), x(ℓ+n)))
```

- 가장 낮은 `d(ℓ)` 찾기 → 가장 유사한 연속 블록 제거
- `keep_last_layer=True`: 마지막 레이어 항상 유지 (필수!)

### B/C Split 정책
```bash
--split_policy half       # 50:50 분할 (기본값)
--split_policy ratio      # 커스텀 비율
--split_ratio 0.3         # 30:70 분할
```

---

## 🔧 주요 파일 설명

| 파일 | 역할 | Mistral 수정 |
|------|------|-------------|
| `identity.py` | PassLayer 정의 | MistralDecoderLayer 지원 추가 |
| `simdrop.py` | Angular distance 계산 | 모델 타입 자동 감지 |
| `layeronly_drop.py` | 메인 프루닝 스크립트 | Mistral config 처리 |
| `bundler.py` | B/C 번들 저장 | model_type 파라미터 추가 |
| `entropy_cli.py` | 동적 라우팅 REPL | Mistral/LLaMA 통합 |
| `data.py` | C4 데이터 로더 | 수정 불필요 (공통) |

---

## 🧪 테스트

```bash
# 1. Pruning 실행 (drop_frac=0)
python -m pruning.layeronly_drop \
  --model mistralai/Mistral-7B-v0.1 \
  --drop_frac 0.0 \
  --save_dir ./test/A

# 2. 모델 타입 확인
python -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('./test/A')
print(f'Model type: {cfg.model_type}')
print(f'Num layers: {cfg.num_hidden_layers}')
"

# 3. Entropy CLI 테스트
python -m pruning.entropy_cli \
  --base_model ./test/A \
  --bundles_dir ./test/bundles \
  --max_new_tokens 32
```

---

## 📈 성능 비교 (논문 기준)

| Model | Original | Pruned (30%) | MMLU Δ |
|-------|----------|--------------|--------|
| Mistral 7B | 32 layers | 22 layers | -2.1% |
| LLaMA-2 7B | 32 layers | 24 layers | -3.4% |
| LLaMA-2 13B | 40 layers | 28 layers | -1.8% |

**Mistral의 장점:**
- 같은 프루닝 비율에서 성능 저하가 더 적음
- GQA 덕분에 inference 속도도 더 빠름

---

## ⚠️ 주의사항

### 1. Learning Rate (중요!)
```bash
# LoRA 학습 시 반드시 Mistral용 LR 사용
--learning_rate 3e-6  # Mistral (NOT 3e-4!)
--lora_r 4            # Mistral (NOT 2!)
```

### 2. Memory 요구사항
- **Pruning**: ~20GB (1x A100/3090)
- **Stage routing**: ~12GB (pruned model)
- **Full model**: ~16GB (all bundles loaded)

### 3. Batch Size
```bash
# Mistral은 LLaMA보다 메모리 효율적
--nsamples 64      # OK for Mistral
--max_batches 32   # 충분함
```

---

## 🔗 참고 자료

- 논문: https://arxiv.org/abs/2403.17887
- Mistral: https://mistral.ai/
- HuggingFace: https://huggingface.co/mistralai

---

## 📝 Citation

```bibtex
@article{gromov2024unreasonable,
  title={The Unreasonable Ineffectiveness of the Deeper Layers},
  author={Gromov, Andrey and Tirumala, Kushal and Shapourian, Hassan and Glorioso, Paolo and Roberts, Daniel A},
  journal={ICLR},
  year={2025}
}
```

---

## 🆚 LLaMA vs Mistral 코드 호환성

기존 LLaMA 코드와 **100% 호환**됩니다:

```bash
# LLaMA-2 (기존 코드 그대로 동작)
python -m pruning.layeronly_drop \
  --model meta-llama/Llama-2-7b-hf \
  ...

# Mistral (동일한 인터페이스)
python -m pruning.layeronly_drop \
  --model mistralai/Mistral-7B-v0.1 \
  ...
```

**자동 감지 로직:**
1. `config.model_type` 확인
2. `config.architectures` 확인
3. `model.model.layers` vs `model.model.decoder.layers` 구조 확인

---

## 🐛 Troubleshooting

### Q: "Cannot import MistralDecoderLayer" 에러
A: transformers 업데이트
```bash
pip install --upgrade transformers>=4.34.0
```

### Q: Bundle restore 시 shape mismatch
A: manifest.json에서 model_type 확인
```bash
cat ./mistral_results/pruning/A/manifest.json | grep model_type
```

### Q: OOM (Out of Memory)
A: 
```bash
--max_batches 16    # 줄이기
--seqlen 512        # 줄이기
--nsamples 32       # 줄이기
```

---

**Happy Pruning! 🎉**