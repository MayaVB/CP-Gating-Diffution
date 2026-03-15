from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="JacobLinCool/VoiceBank-DEMAND-16k",
    repo_type="dataset",
    local_dir="voicebank"
)
