"""从 GLUE 论文 PDF 提取 D_eff / R_eff / Psi_eff 的完整定义与算法步骤。"""
import re
import sys

from pypdf import PdfReader

PDF = "/mnt/mydisk/xinguan-brainai/code/glue_paper.pdf"

r = PdfReader(PDF)
pages = [(r.pages[i].extract_text() or "") for i in range(len(r.pages))]
print(f"PAGES: {len(pages)}\n")

# 找含关键词的页
KEYS = ["anchor point", "D_eff", "Deff", "effective dimension",
        "Gaussianiz", "gaussianiz", "downsample", "Algorithm 1", "contribution ratio"]
for i, t in enumerate(pages):
    hits = [k for k in KEYS if k.lower() in t.lower()]
    if hits:
        print(f"--- page {i+1}: {hits} ---")

# 打印最相关的几页全文
TARGETS = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else []
for p in TARGETS:
    print(f"\n{'='*70}\nPAGE {p}\n{'='*70}")
    print(pages[p-1])
