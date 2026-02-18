#!/usr/bin/env python3
"""
A/B/C 단계별 모델 응답 테스트
- A: 프루닝된 base 모델 (B+C 레이어 제거됨)
- AB: B 레이어 복원, C는 PassLayer
- FULL: B+C 모두 복원

python -m llama_prune_lora.pruning.stage_tester \
  --base_model ./llama2_7b_merged_models/A_merged \
  --bundles_dir ./7b_results/pruning/bundles \
  --prompts "explain France" \
  --stages A,FULL \
  --max_new_tokens 50 \
  --output results.json

"""
import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
except:
    LlamaDecoderLayer = None


class PassLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings=None,
        **kwargs
    ):
        # 혹시 들어오는 hidden_states가 tuple이면 첫 원소만 사용
        if isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]

        # ✅ 핵심: transformers 쪽 래퍼 기대에 맞게 "텐서"를 반환
        return hidden_states


# 레이어 경로 자동 탐지
def _get_layers(model):
    for path in ["model.layers", "model.model.layers", "model.decoder.layers"]:
        try:
            parts = path.split(".")
            obj = model
            for p in parts:
                obj = getattr(obj, p)
            return obj
        except:
            continue
    raise RuntimeError("레이어 경로를 찾을 수 없습니다")


# Bundle 파일 스캔
def _scan_bundles(bundles_dir: Path) -> tuple[Dict[int, Path], Dict[int, Path]]:
    B_map, C_map = {}, {}
    pattern = re.compile(r"layer_(\d+)\.safetensors$")
    
    for bundle_type, target_map in [("B", B_map), ("C", C_map)]:
        bundle_path = bundles_dir / bundle_type
        if bundle_path.exists():
            for f in bundle_path.glob("layer_*.safetensors"):
                if m := pattern.search(f.name):
                    target_map[int(m.group(1))] = f
    
    return B_map, C_map


# state_dict에서 레이어 prefix 제거
def _strip_prefix(sd: Dict[str, torch.Tensor], layer_idx: int) -> Dict[str, torch.Tensor]:
    prefixes = [f"model.layers.{layer_idx}.", f"model.model.layers.{layer_idx}.", 
                f"model.decoder.layers.{layer_idx}.", f"layers.{layer_idx}."]
    result = {}
    for k, v in sd.items():
        new_k = k
        for prefix in prefixes:
            if prefix in new_k:
                new_k = new_k.split(prefix, 1)[1]
                break
        result[new_k] = v
    return result


# Stage Manager
class StageManager:
    def __init__(self, model, bundles_dir: Path, device: str):
        self.model = model
        self.layers = _get_layers(model)
        self.device = torch.device(device)
        self.num_layers = len(self.layers)
        
        # Bundle 스캔
        self.B_map, self.C_map = _scan_bundles(bundles_dir)
        self.removed_idx = sorted(set(self.B_map.keys()) | set(self.C_map.keys()))
        
        # 레이어별 디바이스 저장
        self.layer_devices = {}
        for i, layer in enumerate(self.layers):
            p = next(layer.parameters(), None)
            self.layer_devices[i] = p.device if p is not None else self.device
        
        self.current_pass = set()  # 현재 PassLayer 상태
        print(f"✓ StageManager 초기화: {len(self.removed_idx)}개 레이어 (B:{len(self.B_map)}, C:{len(self.C_map)})")
    
    def _restore_layer(self, idx: int):
        """Bundle에서 레이어 복원"""
        bundle_path = self.B_map.get(idx) or self.C_map.get(idx)
        if not bundle_path:
            raise FileNotFoundError(f"layer_{idx} bundle을 찾을 수 없습니다")
        
        # 새 레이어 생성
        if LlamaDecoderLayer:
            try:
                new_layer = LlamaDecoderLayer(self.model.config, idx)
            except TypeError:
                new_layer = LlamaDecoderLayer(self.model.config)
        else:
            raise RuntimeError("LlamaDecoderLayer import 실패")
        
        # 가중치 로드
        sd = load_file(str(bundle_path), device="cpu")
        sd = _strip_prefix(sd, idx)
        new_layer.load_state_dict(sd, strict=False)
        
        # 디바이스 이동 및 교체
        dev = self.layer_devices[idx]
        new_layer = new_layer.to(dev)
        old = self.layers[idx]
        self.layers[idx] = new_layer
        del old
    
    def _pass_layer(self, idx: int):
        """레이어를 PassLayer로 교체"""
        dev = self.layer_devices[idx]
        old = self.layers[idx]
        self.layers[idx] = PassLayer().to(dev)
        del old
    
    def set_stage(self, stage: str):
        """Stage 전환: A / AB / FULL"""
        stage = stage.upper()
        
        # 각 stage별 pass 대상 결정
        if stage == "A":
            pass_set = set(self.removed_idx)  # B+C 모두 pass
        elif stage == "AB":
            pass_set = set(self.C_map.keys())  # C만 pass
        elif stage == "FULL":
            pass_set = set()  # 모두 restore
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        # 상태 전환
        for idx in self.removed_idx:
            if idx in pass_set and idx not in self.current_pass:
                self._pass_layer(idx)
                self.current_pass.add(idx)
            elif idx not in pass_set and idx in self.current_pass:
                self._restore_layer(idx)
                self.current_pass.discard(idx)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"→ Stage [{stage}] 활성화 (PassLayer: {len(self.current_pass)}개)")


# 텍스트 생성
@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, device: str, max_new: int = 100) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False
    )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="A 모델 경로")
    parser.add_argument("--bundles_dir", required=True, help="bundles/ 디렉토리")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--prompts", nargs="+", default=["Hello, how are you?"], help="테스트 프롬프트")
    parser.add_argument("--stages", default="A,AB,FULL", help="테스트할 stage (쉼표 구분)")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--output", type=str, help="결과 저장 JSON 파일")
    args = parser.parse_args()
    
    # dtype 설정
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    
    # 모델/토크나이저 로드
    print(f"[1/4] 모델 로드: {args.base_model}")
    
    # 토크나이저 로드 (여러 방법 시도)
    base_path = Path(args.base_model).resolve()
    tokenizer_paths = [
        base_path / "original_config",
        base_path,
    ]
    
    tokenizer = None
    for tok_path in tokenizer_paths:
        if not tok_path.exists():
            continue
            
        print(f"  시도: {tok_path}")
        
        # 방법 1: use_fast=False
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(tok_path), use_fast=False)
            print(f"✓ 토크나이저 로드 성공 (slow)")
            break
        except Exception as e1:
            print(f"    slow 실패: {type(e1).__name__}")
        
        # 방법 2: legacy=True
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(tok_path), use_fast=False, legacy=True)
            print(f"✓ 토크나이저 로드 성공 (legacy)")
            break
        except Exception as e2:
            print(f"    legacy 실패: {type(e2).__name__}")
        
        # 방법 3: LlamaTokenizer 직접 로드
        try:
            from transformers import LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(str(tok_path))
            print(f"✓ 토크나이저 로드 성공 (LlamaTokenizer)")
            break
        except Exception as e3:
            print(f"    LlamaTokenizer 실패: {type(e3).__name__}")
    
    if tokenizer is None:
        raise FileNotFoundError(
            f"토크나이저를 찾을 수 없습니다. 모든 방법 실패.\n"
            f"tokenizers 라이브러리 재설치 필요:\n"
            f"  pip install --upgrade transformers tokenizers sentencepiece"
        )
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        str(base_path), torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(args.device)
    model.eval()
    
    # Stage Manager 초기화
    print(f"[2/4] StageManager 초기화")
    mgr = StageManager(model, Path(args.bundles_dir), args.device)
    
    # Stage 리스트 파싱
    stages = [s.strip().upper() for s in args.stages.split(",")]
    
    # 테스트 실행
    print(f"[3/4] {len(args.prompts)}개 프롬프트 × {len(stages)}개 stage 테스트")
    results = []
    
    for stage in stages:
        print(f"\n{'='*80}")
        print(f"[STAGE: {stage}]")
        mgr.set_stage(stage)
        
        for i, prompt in enumerate(args.prompts, 1):
            print(f"\n--- Prompt {i}/{len(args.prompts)} ---")
            print(f"Q: {prompt}")
            
            t0 = time.perf_counter()
            response = generate_text(model, tokenizer, prompt, args.device, args.max_new_tokens)
            elapsed = time.perf_counter() - t0
            
            print(f"A: {response}")
            print(f"⏱ {elapsed:.2f}s")
            
            results.append({
                "stage": stage,
                "prompt": prompt,
                "response": response,
                "time_sec": round(elapsed, 3)
            })
    
    # 결과 저장
    if args.output:
        print(f"\n[4/4] 결과 저장: {args.output}")
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "config": {
                    "base_model": args.base_model,
                    "bundles_dir": args.bundles_dir,
                    "stages": stages,
                    "max_new_tokens": args.max_new_tokens
                },
                "results": results
            }, f, indent=2, ensure_ascii=False)
    
    print("\n✓ 완료")


if __name__ == "__main__":
    main()