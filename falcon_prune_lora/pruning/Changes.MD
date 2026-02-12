# Falcon Instruct 모델 구조 대응 변경 사항

## 아키텍처 차이 요약 (Llama vs Falcon)

| 항목 | Llama | Falcon |
|------|-------|--------|
| 레이어 접근 | `model.model.layers` | `model.transformer.h` |
| 임베딩 | `model.model.embed_tokens` | `model.transformer.word_embeddings` |
| 최종 LayerNorm | `model.model.norm` | `model.transformer.ln_f` |
| state_dict prefix | `model.layers.` | `transformer.h.` |
| KV cache 인자 | `past_key_value` | `layer_past` |
| Alibi position encoding | 없음 | `alibi` 인자 존재 (일부 Falcon) |
| trust_remote_code | 불필요 | **필수** |

## 파일별 변경 내역

### `identity.py`
- `LlamaPassLayer` → `FalconPassLayer`
- forward 시그니처에 `alibi`, `layer_past`, `head_mask` 인자 추가
- `past_key_value` → `layer_past` 변경
- 반환 형식을 Falcon의 `FalconDecoderLayer` 출력 튜플에 맞춤

### `simdrop.py`
- `is_opt` 파라미터 제거 (Falcon 전용이므로 불필요)
- 레이어 접근: `model.model.layers` / `model.model.decoder.layers` → `model.transformer.h`
- `drop_consecutive_layers()` 함수에서 `FalconPassLayer` 사용
- `_get_falcon_layers()` 헬퍼 함수 추가

### `bundler.py`
- `is_opt` 파라미터 제거
- `_get_layers()` 함수가 `model.transformer.h` 반환
- `model_type` 기본값을 `"falcon"`으로 변경
- `export_layer_bundle()`, `export_two_bundles()` 에서 `is_opt` 인자 제거

### `layeronly_drop.py`
- 모델 로드 시 `trust_remote_code=True` 추가 (Falcon 필수)
- `is_opt` 관련 분기 로직 모두 제거
- 레이어 접근 `model.model.layers` → `model.transformer.h`
- 임베딩 디바이스 키: `model.embed_tokens` → `transformer.word_embeddings`
- `build_layers_map()`: prefix를 `"transformer.h."` 로 변경
- manifest의 `arch` 필드를 `"falcon"` 으로 설정
- 기본 모델을 `tiiuae/falcon-7b-instruct` 로 변경

### `data.py`
- 변경 없음 (모델 비의존적 코드)

## 지원 모델
- `tiiuae/falcon-7b-instruct`
- `tiiuae/falcon-40b-instruct`
- `tiiuae/falcon-7b`
- `tiiuae/falcon-40b`
- 기타 HuggingFace Falcon 아키텍처 모델