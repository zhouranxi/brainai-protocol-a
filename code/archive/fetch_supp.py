"""尝试获取 GLUE 论文的 Supplementary Information（bioRxiv）与 demo page。"""
import re
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}

CANDIDATES = [
    "https://www.biorxiv.org/content/10.1101/2024.02.26.582157v2.supplementary-material",
    "https://www.biorxiv.org/content/biorxiv/early/2024/02/26/2024.02.26.582157/DC1/embed/media-1.pdf",
    "https://www.biorxiv.org/content/10.1101/2024.02.26.582157v2/DC1/embed/media-1.pdf",
]


def get(url, timeout=40):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.headers.get("Content-Type", ""), r.geturl()


print("=== 尝试 supplementary-material 页面 ===")
try:
    body, ctype, final = get(CANDIDATES[0])
    print("OK", ctype, len(body), final)
    html = body.decode("utf-8", "ignore")
    # 找附件链接
    links = set(re.findall(r'href="([^"]*(?:media-|suppl|DC\d)[^"]*)"', html))
    for l in sorted(links)[:25]:
        print("  ", l)
except Exception as e:
    print("ERR", e)

print()
print("=== 尝试直接 media-1.pdf ===")
for u in CANDIDATES[1:]:
    try:
        body, ctype, final = get(u)
        print("OK", ctype, len(body))
        if body[:4] == b"%PDF":
            open("glue_supp.pdf", "wb").write(body)
            print("  saved -> glue_supp.pdf")
            break
    except Exception as e:
        print("ERR", u[-40:], e)

print()
print("=== demo page ===")
try:
    body, ctype, final = get("https://cnchou.github.io/papers/GLUE/")
    print("OK", len(body), final)
    html = body.decode("utf-8", "ignore")
    for l in sorted(set(re.findall(r'href="([^"]+)"', html))):
        if any(k in l.lower() for k in ["github", "code", "zip", "pdf", "colab",
                                        "notebook", "drive", "zenodo", "figsh"]):
            print("  ", l)
except Exception as e:
    print("ERR", e)
