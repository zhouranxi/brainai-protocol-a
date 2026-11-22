#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ADR-0031: generate P31-P90 (60 new graded-reveal puzzles).

Templates follow P25's seven categories. Each puzzle: q + g1/g2/g3 + gold.
Difficulty tag: hard = g1 category is broad (many candidates), g2 partial,
g3 decisive. Generation is programmatic from fact tables; gold uniqueness
within category is enforced (no two puzzles share an answer).
"""
import json
from pathlib import Path

OUT = Path("/mnt/mydisk/xinguan-brainai/runs/v2_p31/p31_puzzles.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Fact tables: (id, q, g1, g2, g3, gold, difficulty)
# g1 = category (broad), g2 = structural clue (narrows), g3 = decisive.
P = []


def add(q, g1, g2, g3, gold, diff):
    n = len(P) + 31
    P.append({"id": f"P{n}", "q": q, "g1": g1, "g2": g2, "g3": g3,
              "gold": gold, "difficulty": diff})


Q = "我在想一个答案。线索将逐级给出。"

# ---- 成语 (12) ----
CHENGYU = [
    ("形容速度极快。", "含『风』『电』二字。", "含『掣』字。", "风驰电掣", "medium"),
    ("形容工夫到了家、技艺纯熟。", "与『牛』有关。", "主角是厨师。", "庖丁解牛", "medium"),
    ("形容做事犹豫不决。", "与『棋』或『步』有关。", "『举』字开头。", "举棋不定", "medium"),
    ("形容刻苦自励、发奋图强。", "与『胆』有关。", "主角是春秋越国君主。", "卧薪尝胆", "medium"),
    ("形容说话伤人却表面和善。", "与『刀』有关。", "『笑』字开头。", "笑里藏刀", "medium"),
    ("形容非常吝啬。", "与『毛』有关。", "主角是古代一富翁的绰号意象。", "一毛不拔", "medium"),
    ("形容毫无顾忌胡作非为。", "与『法』有关。", "『无法』开头。", "无法无天", "easy"),
    ("形容名不符实、有名无实。", "与『名』有关。", "含『虚』字。", "徒有虚名", "medium"),
    ("形容双方争执不下、僵持。", "与『眼』或『角』有关。", "是动物身体部位组合。", "针锋相对", "hard"),
    ("形容因图小利而失大局。", "与『芝麻』或『瓜』有关。", "动作是『捡』。", "捡了芝麻丢了西瓜", "hard"),
    ("形容非常荒谬、不合情理。", "与『经』有关。", "『荒』字开头。", "荒诞不经", "medium"),
    ("形容夜以继日地勤奋工作。", "与『日』『夜』有关。", "含『兼』或『继』字。", "夜以继日", "easy"),
]
for g1, g2, g3, gold, diff in CHENGYU:
    add(Q, f"类别线索：{g1}", f"再看一条：{g2}", f"最后一条：{g3}", gold, diff)

# ---- 城市 (10) ----
CITY = [
    ("著名古都。", "两字城市，首字『西』。", "兵马俑所在地。", "西安", "easy"),
    ("省会城市。", "两字城市，首字『武』。", "黄鹤楼所在地。", "武汉", "easy"),
    ("省会城市。", "两字城市，首字『长』。", "岳麓山所在地。", "长沙", "easy"),
    ("历史文化名城。", "两字城市，首字『泉』。", "海上丝绸之路起点之一。", "泉州", "medium"),
]
CITY += [
    ("北方沿海城市。", "两字城市，首字『大』。", "老虎滩所在地。", "大连", "easy"),
    ("高原省会。", "两字城市，首字『昆』。", "滇池所在地。", "昆明", "easy"),
    ("中部省会。", "两字城市，首字『郑』。", "少林寺所在省的省会。", "郑州", "medium"),
    ("西北名城。", "两字城市，首字『兰』。", "黄河穿城而过。", "兰州", "medium"),
    ("东北省会。", "两字城市，首字『长』。", "伪满皇宫所在地（与长沙同首字不同城）。", "长春", "hard"),
    ("江南水乡名城。", "两字城市，首字『苏』。", "园林之城。", "苏州", "easy"),
]
for g1, g2, g3, gold, diff in CITY:
    add(Q, f"类别线索：它是{g1}", f"再看一条：{g2}", f"最后一条：{g3}", gold, diff)

# ---- 动物 (10) ----
ANIMAL = [
    ("它是哺乳动物。", "体型巨大，陆地生活。", "鼻子长。", "大象", "easy"),
    ("它是哺乳动物。", "海洋生活。", "以浮游生物为食、地球最大动物。", "蓝鲸", "medium"),
    ("它是鸟类。", "不会飞。", "新西兰国鸟。", "几维鸟", "medium"),
]
ANIMAL += [
    ("它是哺乳动物。", "生活于澳洲。", "有育儿袋、善跳跃。", "袋鼠", "easy"),
    ("它是爬行动物。", "体型巨大。", "海中最大的爬行动物，含『龟』字。", "棱皮龟", "hard"),
    ("它是昆虫。", "社会性昆虫。", "产蜜。", "蜜蜂", "easy"),
    ("它是鸟类。", "猛禽。", "昼行性、视力极佳、常被驯养狩猎。", "猎隼", "hard"),
    ("它是哺乳动物。", "灵长类以外、树栖。", "以桉树叶为食。", "考拉", "easy"),
    ("它是鱼类。", "淡水大型鱼。", "能洄游、含『鲑』字。", "鲑鱼", "hard"),
    ("它是两栖动物。", "体型较大。", "中国特有、被称为『活化石』。", "大鲵", "hard"),
]
for g1, g2, g3, gold, diff in ANIMAL:
    add(Q, f"类别线索：{g1}", f"再看一条：{g2}", f"最后一条：{g3}", gold, diff)

# ---- 乐器 (8) ----
INSTR = [
    ("它是键盘乐器。", "与三角钢琴相对。", "琴身竖立、占地小。", "立式钢琴", "hard"),
    ("它是弦乐器。", "四根弦。", "交响乐团中音域最高的弦乐。", "小提琴", "easy"),
    ("它是管乐器。", "木管。", "黑色、由哨片发声、交响乐团标配。", "单簧管", "medium"),
    ("它是打击乐器。", "成对使用。", "铜制、交响乐团常用。", "钹", "medium"),
    ("它是中国传统乐器。", "弹拨乐器。", "『十面埋伏』常以此演奏。", "琵琶", "medium"),
    ("它是中国传统乐器。", "拉弦乐器。", "两根弦、京剧伴奏主力。", "京胡", "hard"),
    ("它是管乐器。", "铜管。", "圆环形、常用于圆舞曲。", "圆号", "medium"),
]
INSTR += [
    ("它是弦乐器。", "弹拨。", "西方称『齐特琴家族』、中国民歌伴奏常客。", "扬琴", "hard"),
]
for g1, g2, g3, gold, diff in INSTR:
    add(Q, f"类别线索：{g1}", f"再看一条：{g2}", f"最后一条：{g3}", gold, diff)

# ---- 天文 (8) ----
ASTRO = [
    ("它是太阳系天体。", "行星。", "离太阳最近。", "水星", "easy"),
    ("它是太阳系天体。", "行星。", "有显著光环。", "土星", "easy"),
    ("它是恒星。", "太阳系外。", "离太阳最近的恒星。", "比邻星", "medium"),
    ("它是天文现象。", "与引力有关。", "光无法逃逸。", "黑洞", "easy"),
    ("它是太阳系天体。", "卫星。", "木星最大卫星。", "木卫三", "medium"),
    ("它是天文概念。", "与星系有关。", "我们所在的星系。", "银河系", "easy"),
    ("它是恒星演化的起点。", "由气体尘埃构成。", "恒星育婴室。", "星云", "medium"),
    ("它是太阳系天体。", "小天体。", "有著名彗星命名、周期约 76 年。", "哈雷彗星", "medium"),
]
for g1, g2, g3, gold, diff in ASTRO:
    add(Q, f"类别线索：{g1}", f"再看一条：{g2}", f"最后一条：{g3}", gold, diff)

# ---- 人物 (8) ----
PERSON = [
    ("他是科学家。", "物理学家。", "相对论提出者。", "爱因斯坦", "easy"),
    ("他是科学家。", "生物学家。", "进化论提出者。", "达尔文", "easy"),
    ("她是科学家。", "中国科学家。", "青蒿素发现者。", "屠呦呦", "medium"),
    ("他是数学家。", "中国古代。", "圆周率精确计算、祖率。", "祖冲之", "medium"),
    ("他是发明家。", "美国。", "实用电灯与留声机。", "爱迪生", "easy"),
    ("他是数学家。", "欧洲近代。", "坐标系发明人。", "笛卡尔", "medium"),
    ("她是科学家。", "中国核物理学家。", "与钱三强同为『两弹一星』元勋夫妇、被称为『中国的居里夫人』。", "何泽慧", "hard"),
    ("他是天文学家。", "波兰。", "日心说提出者。", "哥白尼", "easy"),
]
for g1, g2, g3, gold, diff in PERSON:
    add(Q, f"类别线索：{g1}", f"再看一条：{g2}", f"最后一条：{g3}", gold, diff)

# ---- 其他 (4) ----
MISC = [
    ("它是中国地方小吃。", "面食。", "兰州名产（一清二白三红四绿）。", "牛肉面", "medium"),
    ("它是中国传统节日。", "农历。", "五月初五。", "端午节", "easy"),
    ("它是运动器材。", "球拍类。", "拍面有网线、配毛毡面的黄绿色球使用。", "网球拍", "hard"),
    ("它是自然现象。", "与气象有关。", "雷雨前天空中出现、枝状放电发光。", "闪电", "easy")
]
for g1, g2, g3, gold, diff in MISC:
    add(Q, f"类别线索：{g1}", f"再看一条：{g2}", f"最后一条：{g3}", gold, diff)

# ---- 完整性检查 ----
assert len(P) == 60, f"expected 60, got {len(P)}"
golds = [p["gold"] for p in P]
assert len(golds) == len(set(golds)), "duplicate gold inside new set"
old = json.loads(Path("/mnt/mydisk/xinguan-brainai/runs/v2_p25/p25_puzzles.json").read_text(encoding="utf-8"))
old_golds = {p["gold"] for p in old}
overlap = old_golds & set(golds)
assert not overlap, f"overlap with old set: {overlap}"
from collections import Counter
diffs = Counter(p["difficulty"] for p in P)
print("60 puzzles generated. difficulty:", dict(diffs))
OUT.write_text(json.dumps(P, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"written: {OUT}")
