# adaptive_serve.py
import os
import torch

from adaptive_loader import (
    load_stageA_model, load_manifest,
    patch_passlayers, rehydrate_layers_from_bundle
)
from prompt_router import compute_prompt_metrics, PromptDepthRouter

class AdaptiveDepthEngine:
    def __init__(self, A_dir: str, bundles_root: str, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.model, self.tok = load_stageA_model(A_dir, device=device)

        # manifest에서 dropped/B/C 인덱스 확보
        manifest = load_manifest(os.path.join(A_dir, "manifest.json"))
        self.is_opt = (manifest.get("arch", "llama") == "opt")

        self.dropped = list(map(int, manifest["stages"]["A"]["dropped_layers"]))
        self.B_idx = list(map(int, manifest["stages"]["B"]["removed_layers"]))
        self.C_idx = list(map(int, manifest["stages"]["C"]["removed_layers"]))

        # A를 "진짜 A"로 패치 (중요!)
        patch_passlayers(self.model, self.dropped, is_opt=self.is_opt)

        self.bundles_root = bundles_root
        self.loaded = set()  # 이미 복구된 레이어 인덱스 추적

        # 라우터
        self.router = PromptDepthRouter()

        # dtype
        self.dtype = next(self.model.parameters()).dtype

    def ensure_stage(self, stage: str):
        """
        stage=A: 아무것도 안함
        stage=AB: B 레이어 복구
        stage=ABC: B + C 레이어 복구
        """
        if stage in ("AB", "ABC"):
            need_B = [i for i in self.B_idx if i not in self.loaded]
            if need_B:
                rehydrate_layers_from_bundle(
                    self.model,
                    bundle_dir=os.path.join(self.bundles_root, "B"),
                    indices=need_B,
                    is_opt=self.is_opt,
                    device=self.device,
                    dtype=self.dtype,
                    strict=False,  # 환경 따라 strict 문제가 생길 수 있어 기본은 False 권장
                )
                self.loaded.update(need_B)

        if stage == "ABC":
            need_C = [i for i in self.C_idx if i not in self.loaded]
            if need_C:
                rehydrate_layers_from_bundle(
                    self.model,
                    bundle_dir=os.path.join(self.bundles_root, "C"),
                    indices=need_C,
                    is_opt=self.is_opt,
                    device=self.device,
                    dtype=self.dtype,
                    strict=False,
                )
                self.loaded.update(need_C)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7):
        # 1) A로 metrics 계산 (prefill 1회)
        metrics = compute_prompt_metrics(self.model, self.tok, prompt, self.device)
        stage, score = self.router.route(prompt, metrics)

        # 2) 필요 stage까지 복구
        self.ensure_stage(stage)

        # 3) 실제 생성
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **enc,
            do_sample=(temperature > 0),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=False,  # 네 현재 구조에 맞춰 안전하게
            pad_token_id=self.tok.eos_token_id,
        )
        text = self.tok.decode(out[0], skip_special_tokens=True)

        return {
            "stage": stage,
            "score": score,
            "metrics": metrics,
            "text": text,
        }
