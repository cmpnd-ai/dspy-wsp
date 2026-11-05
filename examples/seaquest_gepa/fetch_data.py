"""Download the post's Seaquest files from lm_understand-logs at a pinned commit and check their hashes.

The files stay out of this repo: the Exp 5 prompt contains MinAtar's source (GPL-3.0); the rest is CC BY 4.0.
"""
import hashlib
import urllib.request
from pathlib import Path

BASE = ("https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/"
        "exp5_write_from_scratch/")
FILES = {  # local name: (path in lm_understand-logs, sha256)
    "seaquest_hand.npz": ("networks/seaquest/hand.npz",
                          "2625453048e97b678d73dcf199dfb44d7ef659eb9c94e15263c6c33035fb3e7e"),
    "seaquest_hand_answer.md": ("runs/seaquest/reply/answer.md",
                                "9f4ad41bd23310d97dfac13482504659d1f40be096f48590492cbcfd56756a76"),
    "seaquest_exp5_prompt.txt": ("runs/seaquest/reply/prompt.txt",
                                 "81e51857d7cffac29b12a389dee6ffcda2256a9e03f6cf91f4fd96e1eb855bf6"),
    "exp5_minatar_eval.json": ("eval/minatar_eval.json",
                               "bc10c6f957af95cdd52ee4be085acd249046f6117dbdb8c735fd5ddb2195c47c"),
}

if __name__ == "__main__":
    data = Path(__file__).parent / "data"
    data.mkdir(exist_ok=True)
    for name, (path, sha) in FILES.items():
        f = data / name
        if not (f.exists() and hashlib.sha256(f.read_bytes()).hexdigest() == sha):
            body = urllib.request.urlopen(BASE + path, timeout=60).read()
            assert hashlib.sha256(body).hexdigest() == sha, f"{path}: hash mismatch"
            f.write_bytes(body)
        print(f"ok {name}")
