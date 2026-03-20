#!/usr/bin/env python3
"""
어댑터 생성/부착/머지 검증 스크립트

사용법:
# 1) 어댑터가 어느 레이어에 생성되었는지 확인
python verify_adapter.py check-adapter \
  --adapter_path ./llama_kd_lora_results/adapters/A_lora/stageA/stageA

# 2) 머지 전후 가중치 diff (A 레이어만 변했는지 확인)
python llama_prune_lora/veryfi_adapter.py check-merge \
  --base_dir ./7b_results/pruning/A \
  --merged_dir ./merged_models_llama_7b/A_merged

# 3) 머지 모델의 레이어 구조 확인
python llama_prune_lora/veryfi_adapter.py check-model \
  --model_dir ./merged_models_llama_7b/A_merged
"""

import argparse, json, os, sys
from collections import defaultdict


def cmd_check_adapter(args):
    """어댑터가 어느 레이어에 생성되었는지 확인"""
    adapter_path = args.adapter_path
    print(f"\n{'='*60}")
    print(f"Checking adapter: {adapter_path}")
    print(f"{'='*60}")

    # 1) adapter_config.json
    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        ltt = cfg.get("layers_to_transform", [])
        print(f"\n[adapter_config.json]")
        print(f"  r = {cfg.get('r')}")
        print(f"  lora_alpha = {cfg.get('lora_alpha')}")
        print(f"  target_modules = {cfg.get('target_modules')}")
        print(f"  layers_to_transform = {ltt if ltt else '(없음 = 전체 레이어)'}")
        print(f"  base_model_name_or_path = {cfg.get('base_model_name_or_path', 'N/A')}")
    else:
        print(f"  [warn] adapter_config.json not found")
        ltt = []

    # 2) safetensors 키 분석
    st_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.isfile(st_path):
        # 여러 shard 파일 체크
        st_files = [f for f in os.listdir(adapter_path) if f.endswith(".safetensors")]
        if not st_files:
            print(f"  [warn] No safetensors files found")
            return
        st_path = os.path.join(adapter_path, st_files[0])

    from safetensors import safe_open
    layer_modules = defaultdict(set)
    total_keys = 0

    with safe_open(st_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            total_keys += 1
            # base_model.model.model.layers.{i}.{module}.lora_A.weight
            parts = k.split(".")
            for j, p in enumerate(parts):
                if p == "layers" and j + 1 < len(parts):
                    try:
                        idx = int(parts[j + 1])
                        # module name: 뒤쪽에서 lora_ 앞까지
                        mod_parts = parts[j+2:]
                        mod_name = ".".join(p for p in mod_parts if "lora" not in p and p != "weight" and p != "default")
                        layer_modules[idx].add(mod_name)
                    except ValueError:
                        pass
                    break

    layers_with_lora = sorted(layer_modules.keys())
    print(f"\n[safetensors 분석]")
    print(f"  총 파라미터 키: {total_keys}")
    print(f"  LoRA가 존재하는 레이어: {layers_with_lora}")
    print(f"  레이어 수: {len(layers_with_lora)}")

    if ltt:
        expected = sorted(ltt)
        match = layers_with_lora == expected
        print(f"\n[검증]")
        print(f"  layers_to_transform: {expected}")
        print(f"  실제 LoRA 레이어:    {layers_with_lora}")
        print(f"  일치: {'✓ OK' if match else '✗ MISMATCH'}")
        if not match:
            missing = set(expected) - set(layers_with_lora)
            extra = set(layers_with_lora) - set(expected)
            if missing:
                print(f"  누락: {sorted(missing)}")
            if extra:
                print(f"  초과: {sorted(extra)}")

    print(f"\n[레이어별 모듈]")
    for idx in layers_with_lora[:5]:
        print(f"  layer {idx}: {sorted(layer_modules[idx])}")
    if len(layers_with_lora) > 5:
        print(f"  ... ({len(layers_with_lora) - 5}개 더)")


def cmd_check_merge(args):
    """머지 전후 가중치 diff: A 레이어만 변했는지 확인"""
    print(f"\n{'='*60}")
    print(f"Checking merge diff")
    print(f"  base:   {args.base_dir}")
    print(f"  merged: {args.merged_dir}")
    print(f"{'='*60}")

    from safetensors import safe_open
    import torch

    # manifest에서 인덱스 정보
    man_path = os.path.join(args.base_dir, "manifest.json")
    if not os.path.isfile(man_path):
        man_path = os.path.join(args.merged_dir, "manifest.json")
    if os.path.isfile(man_path):
        with open(man_path) as f:
            man = json.load(f)
        stages = man.get("stages", {})
        A_kept = sorted(stages.get("A", {}).get("kept_layers", []))
        dropped = sorted(stages.get("A", {}).get("dropped_layers", []))
        print(f"\n[manifest]")
        print(f"  A layers: {A_kept}")
        print(f"  dropped:  {dropped}")
    else:
        A_kept, dropped = [], []
        print("\n[manifest] not found")

    # safetensors 파일 찾기
    def find_st_files(d):
        return sorted(f for f in os.listdir(d)
                      if f.endswith(".safetensors") and "adapter" not in f)

    base_files = find_st_files(args.base_dir)
    merged_files = find_st_files(args.merged_dir)

    if not base_files or not merged_files:
        print("[error] safetensors not found in one of the dirs")
        return

    # 키 수집
    def collect_keys(d, files):
        keys = {}
        for f in files:
            with safe_open(os.path.join(d, f), framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    keys[k] = (f, sf.get_tensor(k))
        return keys

    print("\n[Loading base weights...]")
    base_keys = collect_keys(args.base_dir, base_files)
    print(f"  {len(base_keys)} keys")

    print("[Loading merged weights...]")
    merged_keys = collect_keys(args.merged_dir, merged_files)
    print(f"  {len(merged_keys)} keys")

    # diff 분석
    changed_layers = set()
    unchanged_layers = set()
    only_base = set()
    only_merged = set()

    common_keys = set(base_keys.keys()) & set(merged_keys.keys())
    only_base_keys = set(base_keys.keys()) - set(merged_keys.keys())
    only_merged_keys = set(merged_keys.keys()) - set(base_keys.keys())

    print(f"\n[Key 비교]")
    print(f"  공통:     {len(common_keys)}")
    print(f"  base만:   {len(only_base_keys)}")
    print(f"  merged만: {len(only_merged_keys)}")

    # base에만 있는 키 → dropped layer 키일 수 있음
    for k in only_base_keys:
        parts = k.split(".")
        for j, p in enumerate(parts):
            if p == "layers" and j + 1 < len(parts):
                try:
                    only_base.add(int(parts[j + 1]))
                except ValueError:
                    pass
                break

    # 공통 키 diff
    layer_diffs = defaultdict(list)
    for k in sorted(common_keys):
        b_tensor = base_keys[k][1]
        m_tensor = merged_keys[k][1]
        diff = (b_tensor.float() - m_tensor.float()).abs().max().item()

        parts = k.split(".")
        layer_idx = None
        for j, p in enumerate(parts):
            if p == "layers" and j + 1 < len(parts):
                try:
                    layer_idx = int(parts[j + 1])
                except ValueError:
                    pass
                break

        if layer_idx is not None:
            if diff > 1e-6:
                changed_layers.add(layer_idx)
                layer_diffs[layer_idx].append((k.split(".")[-2], diff))
            else:
                unchanged_layers.add(layer_idx)

    print(f"\n[결과]")
    print(f"  변경된 레이어:  {sorted(changed_layers)}")
    print(f"  변경 안된 레이어: {sorted(unchanged_layers - changed_layers)}")
    if only_base:
        print(f"  base에만 존재 (제외됨): {sorted(only_base)}")

    # A 레이어와 비교
    if A_kept:
        A_set = set(A_kept)
        changed_set = changed_layers
        correct_changed = changed_set == A_set
        print(f"\n[검증]")
        print(f"  A 레이어:    {sorted(A_set)}")
        print(f"  변경된 레이어: {sorted(changed_set)}")
        print(f"  A에만 변경됨: {'✓ OK' if correct_changed else '✗ MISMATCH'}")
        if not correct_changed:
            unexpected = changed_set - A_set
            missing = A_set - changed_set
            if unexpected:
                print(f"  A 아닌데 변경: {sorted(unexpected)}")
            if missing:
                print(f"  A인데 미변경:  {sorted(missing)}")

    # 변경된 레이어의 diff 크기
    print(f"\n[레이어별 최대 diff]")
    for idx in sorted(layer_diffs.keys())[:10]:
        max_diff = max(d for _, d in layer_diffs[idx])
        modules = set(m for m, _ in layer_diffs[idx])
        print(f"  layer {idx:2d}: max_diff={max_diff:.6f} modules={sorted(modules)}")
    if len(layer_diffs) > 10:
        print(f"  ... ({len(layer_diffs) - 10}개 더)")


def cmd_check_model(args):
    """머지 모델의 레이어 구조 확인"""
    print(f"\n{'='*60}")
    print(f"Checking model: {args.model_dir}")
    print(f"{'='*60}")

    # config.json
    cfg_path = os.path.join(args.model_dir, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        print(f"\n[config.json]")
        print(f"  model_type:        {cfg.get('model_type')}")
        print(f"  num_hidden_layers: {cfg.get('num_hidden_layers')}")
        print(f"  hidden_size:       {cfg.get('hidden_size')}")

    # manifest.json
    man_path = os.path.join(args.model_dir, "manifest.json")
    if os.path.isfile(man_path):
        with open(man_path) as f:
            man = json.load(f)
        stages = man.get("stages", {})
        print(f"\n[manifest.json]")
        print(f"  A kept:    {stages.get('A', {}).get('kept_layers', [])}")
        print(f"  A dropped: {stages.get('A', {}).get('dropped_layers', [])}")
        print(f"  B:         {stages.get('B', {}).get('removed_layers', [])}")
        print(f"  C:         {stages.get('C', {}).get('removed_layers', [])}")

    # safetensors에서 실제 존재하는 레이어
    from safetensors import safe_open
    present_layers = set()
    total_keys = 0

    for fname in sorted(os.listdir(args.model_dir)):
        if not fname.endswith(".safetensors") or "adapter" in fname:
            continue
        with safe_open(os.path.join(args.model_dir, fname), framework="pt", device="cpu") as f:
            for k in f.keys():
                total_keys += 1
                parts = k.split(".")
                for j, p in enumerate(parts):
                    if p == "layers" and j + 1 < len(parts):
                        try:
                            present_layers.add(int(parts[j + 1]))
                        except ValueError:
                            pass
                        break

    n_layers = cfg.get("num_hidden_layers", "?") if os.path.isfile(cfg_path) else "?"
    print(f"\n[safetensors 분석]")
    print(f"  총 키: {total_keys}")
    print(f"  config 레이어 수: {n_layers}")
    print(f"  실제 가중치 존재 레이어: {sorted(present_layers)}")
    print(f"  가중치 존재 수: {len(present_layers)}")

    if isinstance(n_layers, int):
        missing = set(range(n_layers)) - present_layers
        if missing:
            print(f"  가중치 없는 레이어 (PassLayer/dropped): {sorted(missing)}")
        else:
            print(f"  모든 레이어에 가중치 존재")

    # 파일 크기
    total_bytes = sum(
        os.path.getsize(os.path.join(args.model_dir, f))
        for f in os.listdir(args.model_dir)
        if os.path.isfile(os.path.join(args.model_dir, f))
    )
    print(f"\n[파일 크기]")
    print(f"  총: {total_bytes / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="어댑터/머지 검증 도구")
    sub = parser.add_subparsers(dest="command")

    p1 = sub.add_parser("check-adapter", help="어댑터 레이어 확인")
    p1.add_argument("--adapter_path", required=True)

    p2 = sub.add_parser("check-merge", help="머지 전후 diff")
    p2.add_argument("--base_dir", required=True)
    p2.add_argument("--merged_dir", required=True)

    p3 = sub.add_parser("check-model", help="모델 구조 확인")
    p3.add_argument("--model_dir", required=True)

    args = parser.parse_args()
    if args.command == "check-adapter":
        cmd_check_adapter(args)
    elif args.command == "check-merge":
        cmd_check_merge(args)
    elif args.command == "check-model":
        cmd_check_model(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()