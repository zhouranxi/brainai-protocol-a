"""抓取 Google 表单字段（GLUE early access 申请表）。"""
import re
import urllib.request

URL = "https://docs.google.com/forms/d/e/1FAIpQLSc_IHUkc2zlJv0DIhSL_tiyD7Ty4nCeFdW0U7s-hCVWchefBg/viewform"

req = urllib.request.Request(URL, headers={
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
})
try:
    with urllib.request.urlopen(req, timeout=40) as r:
        html = r.read().decode("utf-8", "ignore")
except Exception as e:
    print("ERR", e)
    raise SystemExit(1)

print(f"len = {len(html)}")

# 表单标题
m = re.search(r'<title>(.*?)</title>', html, re.S)
print("TITLE:", m.group(1).strip()[:200] if m else "?")

# entry id -> 问题文本
entries = dict(re.findall(r'id="entry\.(\d+)"[^>]*>(.*?)</div>', html, re.S))
print(f"\n找到 {len(entries)} 个 entry")
for eid, blk in sorted(entries.items(), key=lambda kv: int(kv[0])):
    # 问题文本在 <div class="...M7eMe"> 内
    q = re.sub(r'<[^>]+>', ' ', blk)
    q = re.sub(r'\s+', ' ', q).strip()
    print(f"  entry.{eid}: {q[:220]}")

# 找 entry id 列表
ids = re.findall(r'name="entry\.(\d+)"', html)
print("\n表单字段 id（去重）:", sorted(set(ids), key=int))

# 找出所有含 required 的标记
reqs = re.findall(r'id="entry\.(\d+)"[^>]*aria-required="true"', html)
print("必填:", sorted(set(reqs), key=int))

# 提取选项（radio / checkbox）
opts = re.findall(r'<div class="[A-Za-z0-9_\- ]*exportItem"[^>]*>(.*?)</div>', html, re.S)
if opts:
    print("\n选项:")
    for o in opts[:30]:
        t = re.sub(r'<[^>]+>', ' ', o)
        print("  -", re.sub(r'\s+', ' ', t).strip()[:100])

# 找提交 URL 与 form id
fm = re.search(r'<form[^>]*action="([^"]+)"', html)
print("\nform action:", fm.group(1) if fm else "?")

# 探测题目标题（可能在 JS 变量里）
for key in ["'desc'", "entry_", "FB_PUBLIC_LOAD_DATA_"]:
    idx = html.find(key)
    if idx > 0:
        print(f"\n--- {key} 附近 ---")
        print(html[idx:idx + 600].replace("\\n", " ")[:600])
        break
