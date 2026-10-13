"""Print the local path of the GGUF the Space serves, downloading it from the model repo once."""

from app.config import space_info
from huggingface_hub import hf_hub_download

if __name__ == "__main__":
    gguf = space_info()["gguf"]
    print(hf_hub_download(repo_id=gguf["repo"], filename=gguf["file"], revision=gguf.get("revision")))
