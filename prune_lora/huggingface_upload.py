from huggingface_hub import HfApi

repo_id = "dddreamerrr/7b_lora_result"   # 업로드할 repo (없으면 생성됨)
local_dir = "./lora_results"           # 업로드할 로컬 폴더

api = HfApi()  # <- token 인자 없음 (CLI 로그인 토큰 자동 사용)

api.create_repo(
    repo_id=repo_id,
    repo_type="dataset",  # LoRA만 올릴 거면 dataset으로 두는 것도 OK
    exist_ok=True,
    private=True,         # 원하면 False
)

api.upload_folder(
    folder_path=local_dir,
    repo_id=repo_id,
    repo_type="dataset",
)
print("Uploaded:", repo_id)
