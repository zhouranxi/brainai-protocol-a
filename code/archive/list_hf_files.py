"""用 HF API 列出数据集仓库文件树，确定正确的 parquet 路径。"""
import json
import sys
import time
import urllib.request

MIRROR = "https://hf-mirror.com"


def api(path, tries=4):
    url = f"{MIRROR}/api/datasets/{path}"
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            print(f"  retry {i+1}: {e}")
            time.sleep(5 * (i + 1))
    return None


def show(repo, filt=None):
    print(f"\n===== {repo} =====")
    d = api(repo)
    if not d:
        print("  API failed")
        return
    for s in d.get("siblings", []):
        n = s.get("rfilename", "")
        if filt and filt not in n:
            continue
        print(f"  {n}")
    print(f"  (total {len(d.get('siblings', []))} files)")


if __name__ == "__main__":
    for repo in ["uoft-cs/cifar100", "uoft-cs/cifar10", "axiong/cifar10-c"]:
        show(repo)
        time.sleep(3)
