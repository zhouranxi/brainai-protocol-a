#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""API probe + G0-L round zero: Qwen on graded-reveal task (text version)."""
import json
import urllib.request
import time

BASE = "http://localhost:8543/v1/chat/completions"


def ask(messages, max_tokens=512, temperature=0.0):
    payload = json.dumps({
        "model": "qwen",
        "messages": messages,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(BASE, data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    msg = d["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    # Qwen3 thinking mode: conclusions live in content; if empty, fall
    # back to reasoning (the thinking trace) so we can still see behavior.
    return (content if content.strip() else "[think] " + reasoning[-400:],
            time.time() - t0)


# ---- 1. health check ----
out, dt = ask([{"role": "user", "content": "回复两个字：正常"}], max_tokens=8)
print("API_OK:", out.strip()[:20], f"({dt:.1f}s)")

# ---- 2. G0-L round zero: graded evidence, text-only task ----
# Task: word-analogy disambiguation with evidence arriving in 3 grades.
# Grade 1 (least evidence): only the category hint.
# Grade 2: + the first letters. Grade 3: + letter count.
# The optimal behavior is to maintain a hypothesis SET early and
# narrow it as evidence arrives - exactly what we measure.

PROMPTS = [
    {
        "q": "我在想一个中文成语。线索将逐级给出。",
        "g1": "类别线索：它与『目光短浅』意思相近。",
        "g2": "再看一条：首字是『鼠』。",
        "g3": "最后一条：四个字，第二、三字是『目寸』。",
        "gold": "鼠目寸光",
    },
    {
        "q": "我在想一个中国城市。线索将逐级给出。",
        "g1": "类别线索：它是长江沿岸的大城市。",
        "g2": "再看一条：三字城市。",
        "g3": "最后一条：简称『宁』。",
        "gold": "南京市",
    },
    {
        "q": "我在想一种动物。线索将逐级给出。",
        "g1": "类别线索：它是海洋哺乳动物，以聪明著称。",
        "g2": "再看一条：使用回声定位。",
        "g3": "最后一条：英文单词首字母 D。",
        "gold": "海豚",
    },
]

for item in PROMPTS:
    print(f"\n=== 谜底：{item['gold']} ===")
    history = [{"role": "user", "content":
                item["q"] + "\n" + item["g1"] +
                "\n请给出：1) 你当前的前三个候选假设（按可能性排序）；"
                "2) 你对每个的把握百分比。"}]
    for grade, key in enumerate(("g1", "g2", "g3"), 1):
        if grade > 1:
            history.append({"role": "user", "content": item[key]})
        out, dt = ask(history, max_tokens=300)
        history.append({"role": "assistant", "content": out})
        first_line = out.strip().split("\n")[0][:80]
        has_gold = item["gold"][:2] in out
        print(f"  G{grade} ({dt:.1f}s) 含谜底前2字={has_gold} | {first_line}")
print("\nG0-L round zero done.")
