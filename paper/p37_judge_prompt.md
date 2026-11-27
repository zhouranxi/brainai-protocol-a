# P37（ADR-0036）判读指令 —— 权威文件，automation 只指向本文件

> 本文件于 2026-11-27 12:50 写入，早于任何 P37 数据产生。以本文件为准。

## 背景

Lance 压下全部论文投稿，实验线继续。A（ADR-0035，M-d 截断行打分位置）已完成判 R-C1。B（ADR-0036 / P37）是稳定性批：三候选格式下 top-1 的采样噪声有多大、哪些测量侧干预能降低它。

学校服务器 xinguan@10.10.52.210（SSH 密钥 ~/.ssh/id_tonelens_xinguan，StrictHostKeyChecking=no，远程 python ~/venvs/dl/bin/python）。2026-11-27 12:44 链式发射三臂，**串行**，同一个 llama.cpp 实例：

| 臂 | 命令 | 行数 | 预计耗时 | 日志 | 结果 | 完成标志 |
|---|---|---|---|---|---|---|
| T0 | `--temp 0.0 --mech c --reps 3` | 540 | ~3.1h | logs/p37_t0.log | runs/v2_p37/p37_c_t00_r3.json | /tmp/p37_c_t00_r3_done |
| T7 | `--temp 0.7 --mech c --reps 3` | 540 | ~3.1h | logs/p37_t7.log | runs/v2_p37/p37_c_t07_r3.json | /tmp/p37_c_t07_r3_done |
| T7-b | `--temp 0.7 --mech b --reps 3` | 540 | ~0.3h | logs/p37_t7b.log | runs/v2_p37/p37_b_t07_r3.json | /tmp/p37_b_t07_r3_done |

项目目录 /mnt/mydisk/xinguan-brainai/。实测 M-c 20.7s/条。预计 **2026-11-27 19:20 左右**三臂全部完成。
服务器上有检查脚本 /tmp/p37_check.py（用法 `~/venvs/dl/bin/python /tmp/p37_check.py`）。

注意：ssh 远程命令含 python -c 时引号会被吃掉，复杂逻辑一律写成文件 scp 到 /tmp/ 再执行；scp 与 ssh 分开单条发射，落盘后核 md5。

## 判据（ADR-0036 §十，发射前修订版，已冻结）

- **S-1 噪声下限**：T0 臂 flip(3) > 10% → **整批无效**，报告后停，不读后续闸门。
- **S-2 温度效应**：T7 flip(3) − T0 flip(3) ≥ 30pp
- **S-3 格式效应**：T7 flip(3) − T7-b flip(3) ≥ 20pp
- **S-4′ top-3 噪声**：T7 臂 top-3 召回的 flip(3) 合并 ≤ 15%，且每个 grade ≤ 20%
- **S-4 多数票**：描述性，不判 PASS/FAIL

分层纪律：所有差值先按 grade 报告，三个 grade 同号才判 PASS；总体值仅描述。flip(k) = 前 k 次采样 top-1 指示不全相同的题目比例。

## 步骤

### 1. 检查完成状态

SSH 检查三个 done 标志与行数（每臂 540）。连接超时重试一次，仍不通则按下面"服务器不可达"处理。日志有 Traceback 则报告详情。

### 2. 未完成

报告进度与 ETA，创建一个对应时间的一次性检查任务（prompt 只需一句：读取并严格执行本文件）。**不要每 30 分钟续建**；下次检查时间定在预计完成后 25 分钟。
若服务器不可达：先区分是 SSH 闪断还是本机 VPN 隧道掉线（RuiJie SSLVPN 网卡 Disconnected 需 Lance 手动重连，本机无法自动恢复）。是 VPN 则报告用户、不再续建。

### 3. 完成

scp 拉取三个 JSON 到 github-repo/results/p37_stability/，核 md5。

### 4. 判读

运行 `PYTHONIOENCODING=utf-8 python github-repo/code/p37_judge.py`（在仓库根目录运行亦可）。**判读脚本是发射前冻结并经六场景自测的，不得修改。** 如它崩溃或报错，如实报告，不要改脚本让它跑通。

先跑一次自测确认环境没变：`python github-repo/code/p37_selftest.py` 必须输出 ALL SELF-TESTS PASS。

### 5. 写入 ADR

判读结果追加到 github-repo/paper/adr_0036.md 末尾新开「十二、判读」一节。**追加式，不改既有任何一行**（§一–§十一）。先按 grade 分层列表，再给总体值（标 descriptive）。S-4 多数票的描述性数字也写进去。

### 6. 机制文（github-repo/paper/mechanism/main.tex）

**不改表 2、图 1、§early、任何 rank 数字、任何既有判决。** 可改的只有：

- **S-1 FAIL**（整批无效）：不改 main.tex，只报告。
- **S-4′ PASS**：在 Limitations 的「Three resamples per cell」段**之后**新增一个 `\textbf{...}` 条目（建议标题 `Output-format sampling noise.`），陈述：三候选格式下 temp 0.7 每题 top-1 翻转率 X%（T7 flip(3)，按 grade），显著高于单答案格式的 Y%（S-3 差 Zpp），而 top-3 召回的翻转率仅 W%；因此 top-1 读出的分辨率受格式限制，top-3 读出不受影响。数字全部取自 p37_judge 的输出。
- **S-4′ NOT DETECTED**：新增同样的条目，但如实写 top-3 召回的翻转率，并说噪声不止影响 top-1。
- S-2/S-3 的结果写进同一条目作为上下文。

编译（bash）：

```bash
export PATH="/c/c/tinytex/bin/windows:$PATH"
export TEXMFHOME=/c/c/tinytex/texmf-var
cd github-repo/paper/mechanism
pdftex -fmt=pdflatex -interaction=nonstopmode main.tex > /tmp/tex1.log 2>&1
bibtex main > /tmp/bib.log 2>&1
pdftex -fmt=pdflatex -interaction=nonstopmode main.tex > /tmp/tex2.log 2>&1
pdftex -fmt=pdflatex -interaction=nonstopmode main.tex > /tmp/tex3.log 2>&1
```

要求：/tmp/tex3.log 中 0 个以 `!` 开头的行、0 处 undefined、main.log 中 0 处 Overfull。用 PyMuPDF 量行坐标核对无溢出：letter 版心右边界 x1≈540.3，阈值 541.5，超出的行数应为 0。**不要靠看渲染图下结论。**

然后运行 `PYTHONIOENCODING=utf-8 python github-repo/code/p35_paper_audit.py`，要求 0 FAIL。若新增的 Limitations 条目让审计里"Three resamples per cell"段的边界检查失败，说明审计脚本把 `\textbf{Unequal output budget.}` 当成了该段的结束锚点——此时把新条目加在 `\textbf{Unequal output budget.}` **之前**（即紧跟在 Three resamples 段之后），并同步更新审计脚本里对应的 block 边界，使其继续保护 Three resamples 段原文不变。

### 7. 提交

git add + commit，commit message 写明各闸门判定与关键数字。push 失败则稍后重试一次，仍失败如实报告（不要走 push_via_api.py 以外的非常规手段）。

### 8. 报告

向用户完整报告：每臂行数与截断率；flip(k) 表（按 grade × 臂）；S-1/S-2/S-3/S-4′ 判定及分层明细；S-4 描述性数字；机制文改了什么；以及一句对"后续实验该用 top-1 还是 top-3"的建议（若 S-4′ PASS）。

### 9. 日志

在 .workbuddy/memory/ 当日日志（2026-11-27.md，不存在则创建 2026-11-28.md）**追加**判读记录。**不要直接改 MEMORY.md**（长期记忆只记结论与约定，由主会话蒸馏）。

## 纪律

- 中途读数不作判读，只有全量算数。
- 任何分组差异先按 grade 分层再定性。
- 判读脚本不得改。审计脚本只在 §6 描述的那种边界情况下可以同步调整，且调整后必须重跑确认 0 FAIL。
