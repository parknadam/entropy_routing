# Mistral LoRA Merge & PPL Evaluation Guide

## 전체 파이프라인 개요

```
layeronly_drop.py          mistral_optimized_kd_lora.py         mistral_merge_adapter.py
───────────────────        ─────────────────────────────        ────────────────────────
Mistral-7B (32 layers)     A model + LoRA training              A model + A LoRA → A_merged
        │                          │                            B bundle + B LoRA → B_merged
        ├─ A model (24 layers      ├─ A_lora adapter            C bundle + C LoRA → C_merged
        │   + 8 PassLayer)         ├─ B_lora adapter
        ├─ B bundle (4 layers)     └─ C_lora adapter
        └─ C bundle (4 layers)
                                                                        │
                                                                        ▼
                                                        mistral_eval_ppl_mergedmodel.py
                                                        ────────────────────────────────
                                                        A_merged + B_merged + C_merged
                                                        → A / AB / FULL stage별 PPL 비교
```

---

## 1. `mistral_merge_adapter.py`

LoRA 어댑터를 base 모델 또는 bundle에 머지합니다.

### 핵심 동작

| Stage | base_model | 입력 | 출력 | 출력 크기 |
|-------|-----------|------|------|----------|
| A | `pruning/A` (config.json 있음) | A model + A LoRA | **전체 모델** (config + tokenizer + safetensors) | ~9GB |
| B | `pruning/bundles/B` (bundle dir) | skeleton + B bundle + B LoRA | **번들만** (layer_XXX.safetensors) | ~1.6GB |
| C | `pruning/bundles/C` (bundle dir) | skeleton + C bundle + C LoRA | **번들만** (layer_XXX.safetensors) | ~1.6GB |

### 자동 감지 로직

- `--base_model`에 `config.json`이 **있으면** → HF 모델 → 전체 모델 저장
- `--base_model`에 `config.json`이 **없고** `layer_*.safetensors`가 있으면 → bundle dir → **자동으로 bundle-only 저장**
  - skeleton 모델은 `adapter_config.json`의 `base_model_name_or_path`에서 자동 결정

### 사용법

```bash
# ─── A 머지 (전체 모델 저장) ───
python -m mistral_prune_lora.pruning.mistral_merge_adapter \
  --base_model ./25_mistral_results/pruning/A \
  --adapter_path ./mistral_kd_lora_results/adapters/A_lora/stageA/stageA \
  --output_dir ./merged_models_mistral_7b/A_merged \
  --device cuda:0

# ─── B 머지 (bundle-only 자동 감지) ───
python -m mistral_prune_lora.pruning.mistral_merge_adapter \
  --base_model ./25_mistral_results/pruning/bundles/B \
  --adapter_path ./mistral_kd_lora_results/adapters/B_lora/stageB \
  --output_dir ./merged_models_mistral_7b/B_merged \
  --device cuda:0

# ─── C 머지 (bundle-only 자동 감지) ───
python -m mistral_prune_lora.pruning.mistral_merge_adapter \
  --base_model ./25_mistral_results/pruning/bundles/C \
  --adapter_path ./mistral_kd_lora_results/adapters/C_lora/stageC \
  --output_dir ./merged_models_mistral_7b/C_merged \
  --device cuda:0
```

### CLI 인자

| 인자 | 필수 | 설명 |
|------|------|------|
| `--base_model` | ✓ | A 모델 디렉터리 또는 B/C bundle 디렉터리 |
| `--adapter_path` | △ | 단일 어댑터 경로 |
| `--adapter_paths` | △ | 다중 어댑터 경로 (순차 머지) |
| `--output_dir` | ✓ | 출력 디렉터리 |
| `--device` | | 디바이스 (기본: `cuda:0`) |
| `--tokenizer_path` | | 토크나이저 경로 오버라이드 |
| `--no_verify` | | 머지 후 검증 스킵 |
| `--save_bundle_only` | | bundle-only 저장 강제 |
| `--save_full_model` | | bundle dir이라도 전체 모델 저장 강제 |

### 출력 디렉터리 구조

**A_merged (전체 모델):**
```
A_merged/
├── config.json
├── generation_config.json
├── manifest.json              ← pruning 정보 (dropped layers)
├── model.safetensors          ← ~9GB (PassLayer 위치는 가중치 없음)
├── tokenizer.json
├── tokenizer.model
├── tokenizer_config.json
└── special_tokens_map.json
```

**B_merged / C_merged (번들):**
```
B_merged/
├── layer_021.safetensors      ← ~400MB each
├── layer_022.safetensors
├── layer_023.safetensors
├── layer_024.safetensors
└── bundle_meta.json           ← {"layer_indices": [21,22,23,24], "merged": true}
```

### PassLayer 복원 (A 머지 시)

`from_pretrained()`는 `config.json`의 `num_hidden_layers=32` 기준으로 32개 MistralDecoderLayer를 전부 생성합니다. safetensors에 가중치가 없는 dropped 레이어(예: 21~28)는 **랜덤 초기화**됩니다.

머지 후 `manifest.json`에서 `stages.A.dropped_layers`를 읽어 해당 인덱스를 `PassLayer`(파라미터 0)로 교체합니다. 이렇게 하면:
- `save_pretrained()` 시 해당 레이어 가중치가 저장되지 않아 **~9GB 유지**
- PassLayer는 hidden_states를 그대로 통과시키는 identity이므로 추론에 영향 없음

### LoRA scope 강제

B/C 어댑터가 전체 레이어에 대한 가중치를 포함하더라도, manifest 기반으로 target 레이어를 결정하여 범위 밖 LoRA 가중치를 0으로 만듭니다.

---

## 2. `mistral_eval_ppl_mergedmodel.py`

머지된 모델의 Perplexity를 평가합니다.

### 핵심 기능

- A_merged 모델에 대해 manifest 기반 PassLayer 자동 설치
- B/C merged bundle을 DynamicStageManager로 주입하여 A/AB/FULL stage별 PPL 비교
- `--model_path`에 bundle dir을 넣으면 자동으로 B/C로 분류
- wikitext2/c4/ptb 데이터셋 또는 커스텀 텍스트 파일 지원
- BatchEncoding/TokenizerWrapper 올바르게 처리 (tokens=0 버그 수정 완료)

### 사용법

```bash
# ─── A-only 평가 ───
python -m mistral_prune_lora.mistral_eval_ppl_mergedmodel \
  --model_path ./merged_models_mistral_7b/A_merged \
  --device cuda:0

# ─── A/AB/FULL 비교 (자동 분류) ───
# model_path에 bundle dir을 같이 넣으면 자동으로 B/C로 분류됩니다
python -m mistral_prune_lora.mistral_eval_ppl_mergedmodel \
  --model_path ./merged_models_mistral_7b/A_merged \
               ./merged_models_mistral_7b/B_merged \
               ./merged_models_mistral_7b/C_merged \
  --device cuda:0

# ─── A/AB/FULL 비교 (명시적) ───
python -m mistral_prune_lora.mistral_eval_ppl_mergedmodel \
  --model_path ./merged_models_mistral_7b/A_merged \
  --b_bundle ./merged_models_mistral_7b/B_merged \
  --c_bundle ./merged_models_mistral_7b/C_merged \
  --stages A,AB,FULL \
  --device cuda:0

# ─── 원본 모델과 비교 ───
python -m mistral_prune_lora.mistral_eval_ppl_mergedmodel \
  --model_path mistralai/Mistral-7B-v0.1 ./merged_models_mistral_7b/A_merged \
  --tokenizer_path mistralai/Mistral-7B-v0.1 \
  --device cuda:0

# ─── 구버전 bundles_dir 호환 ───
python -m mistral_prune_lora.mistral_eval_ppl_mergedmodel \
  --model_path ./merged_models_mistral_7b/A_merged \
  --bundles_dir ./25_mistral_results/pruning/bundles \
  --stages A,AB,FULL \
  --device cuda:0

# ─── 텍스트 파일로 평가 ───
python -m mistral_prune_lora.mistral_eval_ppl_mergedmodel \
  --model_path ./merged_models_mistral_7b/A_merged \
  --text_file ./eval_corpus.txt \
  --device cuda:0
```

### CLI 인자

| 인자 | 필수 | 설명 |
|------|------|------|
| `--model_path` | ✓ | 전체 모델 경로 (여러 개 가능). bundle dir은 자동 분류됨 |
| `--device` | | 디바이스 (기본: `cuda:0`) |
| `--dtype` | | `bf16` / `fp16` / `fp32` (기본: `bf16`) |
| `--dataset` | | 데이터셋 (기본: `wikitext2`) |
| `--split` | | 평가 split (기본: `test`) |
| `--seqlen` | | 시퀀스 길이 (기본: `2048`) |
| `--batch_size` | | 배치 크기 (기본: `1`) |
| `--nsamples` | | 샘플 수 (기본: `256`) |
| `--seed` | | 랜덤 시드 (기본: `0`) |
| `--max_batches` | | 최대 배치 수 (기본: 전부) |
| `--text_file` | | 커스텀 텍스트 파일 경로 |
| `--b_bundle` | | 머지된 B 번들 디렉터리 |
| `--c_bundle` | | 머지된 C 번들 디렉터리 |
| `--bundles_dir` | | 구버전 호환: `bundles_dir/B`, `bundles_dir/C` 구조 |
| `--stages` | | 평가 stage (기본: bundle 있으면 `A,AB,FULL`, 없으면 `A`) |
| `--tokenizer_path` | | 토크나이저 오버라이드 |
| `--lora_paths` | | LoRA 미머지 비교용 어댑터 경로 |

### `--model_path` 자동 분류

`--model_path`에 여러 경로를 넣으면 자동으로 분류합니다:

| 조건 | 판별 | 역할 |
|------|------|------|
| `config.json` 있음 | HF 모델 | base 모델로 사용 (로드 + PassLayer 설치) |
| `config.json` 없고 `layer_*.safetensors` 있음 | bundle | 이름에서 B/C 추론 후 자동 할당 |

판별 기준:
- 파일명에 `b_merged` 포함 → B bundle
- 파일명에 `c_merged` 포함 → C bundle

### Stage별 동작

| Stage | 설명 | 활성 레이어 |
|-------|------|-----------|
| A | B/C 모두 PassLayer | kept 레이어만 (예: 24개) |
| AB | B 복구, C만 PassLayer | kept + B (예: 28개) |
| FULL | B+C 모두 복구 | 전체 32개 |

`DynamicStageManager`가 모델을 1회 로드한 후 stage를 전환하며 B/C bundle의 layer safetensors를 `MistralDecoderLayer`로 주입합니다.

### 예상 출력

```
======================================================================
Merged Model PPL Evaluation
======================================================================
Models:      ['./merged_models_mistral_7b/A_merged']
B bundle:    merged_models_mistral_7b/B_merged
C bundle:    merged_models_mistral_7b/C_merged
Stages:      ['A', 'AB', 'FULL']
Dataset:     wikitext2 (split=test)
Seq length:  2048
Device:      cuda:0 (bf16)
======================================================================

────────────────────────────────────────────────────────────
[1/1] ./merged_models_mistral_7b/A_merged
────────────────────────────────────────────────────────────

  ✓ PassLayer 설치: [21, 22, 23, 24, 25, 26, 27, 28] (8개 레이어)
  ✓ mistral | 32 layers (24 active + 8 PassLayer) | 4827.8M params | cuda:0

  [A] Evaluating PPL...
  ┌─────────────────────────────────────────┐
  │ A_merged[A]                             │
  ├─────────────────────────────────────────┤
  │ PPL      = 15.7910                      │
  │ Mean NLL = 2.759441                     │
  │ Tokens   = 333661                       │
  └─────────────────────────────────────────┘

  [AB] Restoring bundle layers...
  ✓ Stage AB: 28 active layers

  [AB] Evaluating PPL...
  ┌─────────────────────────────────────────┐
  │ A_merged[AB]                            │
  ├─────────────────────────────────────────┤
  │ PPL      = 12.3456                      │
  │ Mean NLL = 2.512345                     │
  │ Tokens   = 333661                       │
  └─────────────────────────────────────────┘

  [FULL] Restoring bundle layers...
  ✓ Stage FULL: 32 active layers

  [FULL] Evaluating PPL...
  ┌─────────────────────────────────────────┐
  │ A_merged[ABC]                           │
  ├─────────────────────────────────────────┤
  │ PPL      = 10.7890                      │
  │ Mean NLL = 2.378901                     │
  │ Tokens   = 333661                       │
  └─────────────────────────────────────────┘

======================================================================
Summary
======================================================================
Label                                          PPL     Mean NLL     Tokens
───────────────────────────────────────── ──────────── ──────────── ──────────
A_merged[A]                                   15.7910     2.759441     333661
A_merged[AB]                                  12.3456     2.512345     333661
A_merged[ABC]                                 10.7890     2.378901     333661
======================================================================
```

---

## 3. 해결된 버그들

### 3.1 토크나이저 `./` 상대경로 에러

**증상**: `./merged_models/A_merged`를 tokenizer에 넘기면 HF Hub repo ID로 해석 → OSError

**원인**: transformers가 `./`로 시작하는 경로를 HF Hub repo로 인식

**수정**: `os.path.abspath(path)` 로 절대경로 변환 후 전달

### 3.2 13GB 모델 크기 (PassLayer 미복원)

**증상**: A model(9GB) + A LoRA(180MB) 머지 결과가 13GB

**원인**: `from_pretrained()`가 config의 32 layers 기준으로 전체 MistralDecoderLayer 생성 → dropped 레이어에 랜덤 가중치 할당 → 전부 저장됨

**수정**: 머지 후 `manifest.json`에서 dropped layers를 읽어 `PassLayer`(파라미터 0)로 교체 → 해당 위치 가중치 미저장

### 3.3 Tokens=0 (PPL 평가 실패)

**증상**: PPL = nan, Tokens = 0

**원인 1**: `get_loaders` 호출 시그니처 불일치
```python
# 버그: 두 번째 positional로 tok이 들어가 nsamples=tok이 됨
GET_LOADERS(dataset, tok, seqlen=..., nsamples=...)

# 수정: keyword-only로 호출
GET_LOADERS(dataset, nsamples=256, seed=0, seqlen=2048, tokenizer=tok)
```

**원인 2**: `BatchEncoding`은 `dict`를 상속하여 `iter()` 시 키 문자열 반환
```python
testenc = tokenizer("...", return_tensors='pt')  # BatchEncoding
list(iter(testenc))  # → ["input_ids", "attention_mask"] ← 문자열!
```

**수정**: `iter()` 전에 `.input_ids` 속성 / `dict["input_ids"]` 키로 직접 텐서 추출

### 3.4 C bundle 13GB (bundle-only 미적용)

**증상**: B merged는 400MB×4인데 C merged는 13GB

**원인**: bundle dir를 base_model로 주었는데 `save_bundle_only` 기본값이 `False` → 전체 32 layer 저장

**수정**: `base_model`이 bundle dir(config.json 없음 + layer_*.safetensors 있음)이면 자동으로 `save_bundle_only=True`

---

## 4. 주요 설계 결정

### 토크나이저 폴백 체인 (우선순위)

1. `--tokenizer_path` (사용자 명시)
2. `base_model_path` 자체
3. `base_model_path/original_config/`
4. `adapter_config.json` → `base_model_name_or_path`
5. `manifest.json` → `base_model`

### 모든 로컬 경로에 `os.path.abspath()` 적용

transformers의 `./` 경로 HF Hub 오인 버그 방지를 위해, `os.path.exists(path)`로 로컬 경로 확인 후 `os.path.abspath()`로 변환합니다.

### manifest.json 보존

merge 시 `manifest.json`을 출력에 복사하여, 이후 eval 스크립트가 dropped layers를 읽을 수 있도록 합니다.

---

## 5. 파일 위치

```
mistral_prune_lora/
├── pruning/
│   ├── mistral_merge_adapter.py    ← 머지 스크립트
│   ├── layeronly_drop.py           ← 프루닝 (A model + B/C bundles 생성)
│   ├── identity.py                 ← PassLayer 정의
│   ├── data.py                     ← get_loaders (wikitext2/c4/ptb)
│   └── ...
├── mistral_eval_ppl_mergedmodel.py ← PPL 평가 스크립트
├── mistral_optimized_kd_lora.py    ← LoRA 학습
└── ...
```