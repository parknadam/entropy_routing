# Qwen 모델용 Layer Pruning 가이드

## 개요

LLaMA용으로 작성된 layer pruning 코드를 Qwen 모델에서 사용할 수 있도록 변환한 버전입니다.

## 주요 변경사항

### 1. 모델 아키텍처 차이

| 구분 | LLaMA | Qwen |
|------|-------|------|
| 레이어 경로 | `model.model.layers` | `model.transformer.h` |
| 레이어 클래스 | `LlamaDecoderLayer` | `QWenBlock` |
| 임베딩 | `model.embed_tokens` | `transformer.wte` |
| 레이어 prefix | `model.layers.` | `transformer.h.` |

### 2. 구현된 파일들

```
qwen_identity.py          # Qwen용 PassLayer 구현
qwen_simdrop.py          # Angular distance 계산 및 레이어 선택
qwen_bundler.py          # B/C 번들 저장 로직
qwen_layeronly_drop_final.py  # 통합 메인 스크립트
```

## 사용 방법

### 기본 사용

```bash
python -m qwen_prune_lora.pruning.qwen_layeronly_drop \
  --model Qwen/Qwen-7B \
  --device cuda:0 \
  --drop_frac 0.25 \
  --keep_last_layer \
  --save_dir ./qwen_results/pruning/A \
  --save_removed_dir ./qwen_results/pruning/bundles
```

### Qwen-14B 사용

```bash
python qwen_layeronly_drop_final.py \
  --model Qwen/Qwen-14B \
  --device cuda:0 \
  --drop_frac 0.30 \
  --keep_last_layer \
  --nsamples 64 \
  --seqlen 2048 \
  --max_batches 32 \
  --save_dir ./qwen_14b_results/pruning/A \
  --save_removed_dir ./qwen_14b_results/pruning/bundles
```

### 주요 파라미터 설명

- `--model`: Qwen 모델 경로 (HuggingFace ID 또는 로컬 경로)
- `--device`: 사용할 디바이스 (cuda:0, cuda:1 등)
- `--drop_frac`: 제거할 레이어 비율 (0.0 ~ 1.0)
- `--keep_last_layer`: 마지막 레이어 보존 여부 (권장)
- `--nsamples`: 캘리브레이션 샘플 수
- `--seqlen`: 시퀀스 길이
- `--max_batches`: 최대 배치 수
- `--split_policy`: B/C 분할 방식 (half 또는 ratio)
- `--pass_return_tuple`: PassLayer 반환 형식 (기본 True)

## 출력 구조

```
qwen_results/
├── pruning/
│   ├── A/                          # 프루닝된 모델 (A)
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer_config.json
│   │   ├── manifest.json           # 프루닝 정보
│   │   ├── prune_log.json          # 상세 로그
│   │   ├── layers_map.json         # 레이어-키 매핑
│   │   └── original_config/        # 원본 모델 설정
│   └── bundles/
│       ├── B/                      # 제거된 레이어 (전반부)
│       │   ├── bundle_meta.json
│       │   ├── layer_008.safetensors
│       │   ├── layer_009.safetensors
│       │   └── ...
│       └── C/                      # 제거된 레이어 (후반부)
│           ├── bundle_meta.json
│           ├── layer_012.safetensors
│           └── ...
```

## PassLayer 동작 원리

### QwenPassLayer

```python
class QwenPassLayer(nn.Module):
    def forward(self, hidden_states, **kwargs):
        # hidden_states를 그대로 통과
        # Qwen의 forward signature와 호환
        outputs = (hidden_states,)
        if use_cache:
            outputs = outputs + (layer_past,)
        return outputs
```

### 반환 형식 선택

- `--pass_return_tuple` (기본): tuple 반환 `(hidden_states, kv_cache)`
- `--no_pass_return_tuple`: tensor만 반환 `hidden_states`

일반적으로 tuple 반환이 안전하지만, 오류 발생시 tensor 반환으로 변경하세요.

## 다음 단계: LoRA 어댑터 학습

프루닝 후 각 스테이지별로 LoRA 어댑터를 학습합니다:

### Stage A (A + B,C pass)
```bash
# merged_A 생성
python train_lora_stage_a.py \
  --base_model ./qwen_results/pruning/A \
  --bundles_dir ./qwen_results/pruning/bundles \
  --output_dir ./qwen_results/adapters/A_lora
```

### Stage B (merged_A + B + C pass)
```bash
# merged_B 생성
python train_lora_stage_b.py \
  --base_model ./qwen_results/pruning/A \
  --bundles_dir ./qwen_results/pruning/bundles \
  --a_adapter ./qwen_results/adapters/A_lora/merged \
  --output_dir ./qwen_results/adapters/B_lora
```

### Stage C (merged_A + merged_B + C)
```bash
# merged_C (최종) 생성
python train_lora_stage_c.py \
  --base_model ./qwen_results/pruning/A \
  --bundles_dir ./qwen_results/pruning/bundles \
  --a_adapter ./qwen_results/adapters/A_lora/merged \
  --b_adapter ./qwen_results/adapters/B_lora/merged \
  --output_dir ./qwen_results/adapters/C_lora
```

## 주의사항

### 1. trust_remote_code 필수

Qwen 모델은 `trust_remote_code=True` 옵션이 필요합니다:

```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B",
    trust_remote_code=True  # 필수!
)
```

### 2. 토크나이저 설정

Qwen 토크나이저는 기본적으로 `pad_token`이 없을 수 있습니다:

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### 3. PassLayer 호환성

Qwen의 forward signature가 transformers 버전에 따라 다를 수 있습니다.
오류 발생시 `--no_pass_return_tuple` 옵션을 시도하세요.

### 4. 메모리 사용량

- Qwen-7B: 약 14GB VRAM (fp16)
- Qwen-14B: 약 28GB VRAM (fp16)
- 더 큰 모델은 multi-GPU 또는 quantization 필요

## 검증

프루닝이 올바르게 수행되었는지 확인:

```python
from transformers import AutoModelForCausalLM

# A 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "./qwen_results/pruning/A",
    trust_remote_code=True
)

# 레이어 수 확인
print(f"Layers: {len(model.transformer.h)}")

# manifest 확인
import json
with open("./qwen_results/pruning/A/manifest.json") as f:
    manifest = json.load(f)
print(f"Removed: {manifest['counts']['removed_total']} layers")
print(f"B: {manifest['counts']['B']} layers")
print(f"C: {manifest['counts']['C']} layers")
```

## 문제 해결

### PassLayer 오류

```
TypeError: forward() got an unexpected keyword argument
```

**해결**: `--no_pass_return_tuple` 옵션 사용

### 메모리 부족

```
RuntimeError: CUDA out of memory
```

**해결**:
1. `--seqlen`을 줄이기 (예: 2048 → 1024)
2. `--nsamples`를 줄이기 (예: 64 → 32)
3. Quantization 사용 (코드 수정 필요)

### Bundle 저장 오류

```
AssertionError: B/C overlap on layers
```

**해결**: 이는 내부 버그입니다. `--split_policy ratio --split_ratio 0.5`로 시도

## 참고 자료

- 원본 논문: "The Unreasonable Ineffectiveness of the Deeper Layers" (ICLR 2025)
- Qwen 모델: https://github.com/QwenLM/Qwen
- 논문 코드: https://github.com/tml-epfl/llm-layer-dropping

## 라이선스

원본 코드의 라이선스를 따릅니다.