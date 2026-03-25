import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# =========================
# 설정
# =========================
LOCAL_ADAPTER_DIR = "/workspace/entropy_routing/modified_falcon_kd_lora_results/adapters"
REPO_ID = "dddreamerrr/falcon-7b-total-kd-lora"   
PRIVATE = False
COMMIT_MESSAGE = "Upload PEFT adapter folder"

# HF 토큰: 미리 환경변수로 넣어두는 걸 권장
# export HF_TOKEN=hf_xxx
HF_TOKEN = os.getenv("HF_TOKEN")

def main():
    local_dir = Path(LOCAL_ADAPTER_DIR)
    if not local_dir.exists() or not local_dir.is_dir():
        raise FileNotFoundError(f"어댑터 폴더를 찾을 수 없습니다: {local_dir}")

    if HF_TOKEN is None:
        raise EnvironmentError("HF_TOKEN 환경변수가 없습니다. 먼저 export HF_TOKEN=... 해주세요.")

    # 1) repo 생성 (이미 있으면 통과)
    create_repo(
        repo_id=REPO_ID,
        private=PRIVATE,
        exist_ok=True,
        token=HF_TOKEN,
    )

    # 2) 폴더 업로드
    api = HfApi(token=HF_TOKEN)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message=COMMIT_MESSAGE,
        # 필요하면 아래처럼 필터 가능
        # allow_patterns=["adapter_config.json", "adapter_model.safetensors", "*.json", "*.md"],
        # ignore_patterns=["*.pt", "*.bin", "checkpoint-*", "__pycache__/*"],
    )

    print(f"업로드 완료: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()