import json
import urllib.request

print("=== PyPI 'glue' 是什么 ===")
try:
    with urllib.request.urlopen("https://pypi.org/pypi/glue/json", timeout=30) as r:
        d = json.load(r)
    i = d["info"]
    print("summary :", i.get("summary"))
    print("author  :", i.get("author"))
    urls = i.get("project_urls") or {}
    for k, v in list(urls.items())[:5]:
        print(f"  {k}: {v}")
except Exception as e:
    print("ERR", e)

print()
print("=== GitHub 搜索 GLUE manifold 实现 ===")
q = "glue+manifold+geometry+neural+representation"
try:
    url = ("https://api.github.com/search/repositories?q="
           "glue+untangling+efficiency&per_page=10")
    req = urllib.request.Request(url, headers={"User-Agent": "research"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)
    for it in d.get("items", [])[:10]:
        print(it["full_name"], "|", (it.get("description") or "")[:70])
except Exception as e:
    print("ERR", e)

print()
print("=== 候选：chung-neuroai-lab 下的所有仓库 ===")
try:
    url = "https://api.github.com/orgs/chung-neuroai-lab/repos?per_page=100"
    req = urllib.request.Request(url, headers={"User-Agent": "research"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)
    for it in d:
        print(it["name"], "|", (it.get("description") or "")[:75])
except Exception as e:
    print("ERR", e)
