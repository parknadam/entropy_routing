from huggingface_hub import snapshot_download

repo_id = "dddreamerrr/kd_lora_results"          # <- 바꿔
local_dir = "./kd_lora_results"       # <- 원하는 저장 경로

path = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # 진짜 파일로 복사(환경에 따라 symlink 이슈 방지)
)
print("Downloaded to:", path)
