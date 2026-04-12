original_model_dir / original_N: 원본 모델 HF 경로와 총 레이어 수
A/B/C_dir: pruning 결과 (bundle_meta.json 위치)
A/B/C_merged_dir: LoRA 병합 완료된 state_dict 경로
B_indices / C_indices: 빈 배열이면 bundle_meta.json에서 자동 탐지
eval: PPL stride 512, zero-shot 7 tasks, system benchmark 설정