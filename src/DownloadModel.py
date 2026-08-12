from huggingface_hub import snapshot_download

model_path = snapshot_download(
    repo_id="Qwen/Qwen3-1.7B",
    local_dir="./Qwen3-1.7B"
)

print(model_path)