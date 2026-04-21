from huggingface_hub import snapshot_download

repo_id = "dddreamerrr/eval_results"          # <- 바꿔
local_dir = "./eval_results"       # <- 원하는 저장 경로

path = snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=local_dir,
)
print("Downloaded to:", path)
