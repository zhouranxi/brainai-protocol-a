# P35 判读任务 · 权威 prompt（automation 每次运行时读取本文件并严格按其执行）

> 本文件由 04:45 首次检查会话写入并经逐字校验。automation 的 prompt 字段在传输中出现过多处字符损坏
> （路径、环境变量名、阈值被改），因此所有指令以本文件为准，不以 prompt 字段为准。

P35（ADR-0034，P31 确认批全文重跑 + M-d 截断解除臂）实验判读任务。工作区 C:/Users/Lance/WorkBuddy/2026-10-25-21-15-17。

## 重试背景

04:45 首次检查时校园网 VPN（RuiJie SSLVPN 虚拟网卡）处于 Disconnected，10.10.52.210 完全不可达（ping 与 TCP 22 均 timeout）。
先确认可达性再做判读。若仍不可达，按步骤 2 处理，重试不超过 2 次。

## 背景

学校服务器 xinguan@10.10.52.210（SSH 密钥 ~/.ssh/id_tonelens_xinguan，StrictHostKeyChecking=no，远程 python ~/venvs/dl/bin/python）。
2026-11-25 00:05 链式发射两个串行任务：

- **R1200**：`code/p35_race_fulltext.py --max-tokens 1200 --mechs abcd`，2160 条（60 题 × 4 机制 × 3 级 × 3 重采样）。
  日志 /mnt/mydisk/xinguan-brainai/runs/v2_p35/ 上一级 logs/p35_1200.log，结果 runs/v2_p35/p35_race_1200.json，完成标志 /tmp/p35_1200_done。
  **实测速度**（05:05 核对，819 条/5h）：M-a 12.9s、M-b 1.7s、M-c 20.8s、M-d 55.6s 每条（M-d = base + 批判两次调用）。
  **预计完成 2026-11-25 13:30 左右**。原估"约 3h"是错的。
- **R2400**：`--max-tokens 2400 --mechs d`，540 条，R1200 结束后自动启动。
  M-d 每条 ≥56s，2400 token 更慢估 ~70s → **预计 10.5h，2026-11-26 00:00 左右完成**。原估"约 1.5h"是错的。
  日志 logs/p35_2400.log，结果 p35_race_2400.json，完成标志 /tmp/p35_2400_done。

每行记录含 text（全文）、finish_reason（API 原生截断标志，"length" 表示截断）；M-d 另有 base 与 base_finish_reason。
服务器上有检查脚本 /tmp/p35_check.py（用法：`~/venvs/dl/bin/python /tmp/p35_check.py 1200`）。

注意：ssh 远程命令含 python -c 时引号会被吃掉，复杂逻辑一律写成文件 scp 到 /tmp/ 再执行。

## 步骤

### 1. 检查完成状态

SSH 检查两个 done 标志与 JSON 行数（R1200 应为 2160，R2400 应为 540）。
连接超时重试一次，仍不通则报告用户。若日志有 Traceback，报告详情。

### 2. 未完成分支（两阶段判读）

**阶段 A（R1200 完成、R2400 未完成）**：这是预期中的正常中间态（13:30–24:00 之间）。
拉取 p35_race_1200.json 到 github-repo/results/p35_fulltext/，执行步骤 4 的 **Q1 部分**与步骤 4 末尾的 **一致性检查**，
把 Q1 结果追加写入 adr_0034.md §六，按步骤 6 的"Q1 部分"改机制文，commit。然后为 R2400 创建一次性检查任务，
**定在 2026-11-26 00:30**（不要每 30 分钟续建）。新任务 prompt 只需一句：读取并严格执行本文件。

**R1200 未完成**：报进度与 ETA。若预计 1h 内完成，创建对应时间的一次性检查任务；否则报告用户不再续建。
若服务器不可达：报告诊断结论（VPN 隧道断开需用户手动重连，本机无法自动恢复），不再创建新的重试任务。

**阶段 B（R2400 完成）**：执行步骤 3–7 全部。

### 3. 拉取结果

若完成：scp 拉取两个 JSON 到 github-repo/results/p35_fulltext/。

### 4. 预注册判读（github-repo/paper/adr_0034.md）

**Q1（描述性，无闸门）**
- R1200 各机制 finish_reason == "length" 的比例，与 batch-1 对照（M-d 57.8%、M-c 7.8%、M-a 4.4%、M-b 0%）。
- 另报告 M-d 的 base_finish_reason 分布。

**Q2（有闸门）**
- M-d top-1（rank == 1 的比例）在 G1/G2/G3 上 R2400 vs R1200 的差值。
- 两种口径都报：(a) 按 puzzle 取 3 rep 的中位 rank 后统计；(b) 直接按行统计。
- R-T1（截断有行为影响）：任一 grade 差值 ≥ 10pp 且 sign test p < 0.05。sign test 按题配对：同一 puzzle、同一 grade 的 M-d top-1 指示，R2400 vs R1200。
- R-T0（截断无行为影响）：三 grade 差值均 < 10pp，或 p ≥ 0.05。
- 次要：R2400 的 dup 分布 vs R1200；R2400 的 finish_reason 分布（预期 length 比例大幅下降）。
- 额外一致性检查：R1200 重跑的各机制 G3 top-1 与原 P31（results/p31_confirm/p31_race.json）对比，应同量级（temp 0.7 有随机性，不要求逐条相同）。

### 5. 写入 ADR

判读结果追加写入 github-repo/paper/adr_0034.md 的「六」节，git add + commit。

### 6. 修订机制文 github-repo/paper/mechanism/main.tex

**阶段 A（仅 Q1，R1200 完成时）**：
- 测量伪影节 "Residual truncation, disclosed" 段末尾的 "The confirmation batch used the same cap and did not archive full texts, so the census could not be repeated there." 改为确认批实测：
  "A re-run of the confirmation batch with full texts archived (same cap, same prompts) gives N\% for \Mcrit{}, against X\% for \Mmaint{} and Y\% for \Mstate{}; the pre-critique candidate step was cut in only Z\% of \Mcrit{} calls, so the cut falls on the critique pass itself."（N/X/Y/Z 用实测值）
- Reproducibility 段 "no full texts" 改为 "full texts archived"，并补一句 re-run 批在 results/p35_fulltext/。
- Limitations 的 "Unequal output budget" 段**本阶段不改**——它说"confirmation batch's full texts were not retained to repeat the census"，等阶段 B 一起改。
- 一致性检查若通过（R1200 各机制 G3 top-1 与原 P31 同量级，差 < 10pp），不改正文；若某机制差 ≥ 10pp，在报告里标红，不动 main.tex。

**阶段 B（Q2 判定后）**：
- **R-T0 分支**：
  - Limitations 的 "Unequal output budget" 段改写为：确认批已重跑存全文，截断率 X%，解除截断（2400 token）后 M-d 行为无显著变化（三 grade 差值均 < 10pp）。
  - 测量伪影节 "Residual truncation, disclosed" 段末尾加一句：lifting the cap to 2,400 tokens leaves \Mcrit{}'s top-1 unchanged at every grade (largest difference D\,pp), so the truncation is a measurement artifact, not a behavioral confound.
  - Reproducibility 已在阶段 A 更正，核对即可。
- **R-T1 分支**：
  - 在 \section{The measurement artifact} 末尾加一段，说明哪个 grade 受影响、幅度、方向。
  - Limitations 保留 Unequal output budget，并注明已验证为行为混杂；把 "were not retained" 改为实测截断率。
  - 不要自行重判 R2。

**编译**（bash）：

```bash
export PATH="/c/c/tinytex/bin/windows:$PATH"
export TEXMFHOME=/c/c/tinytex/texmf-var
cd github-repo/paper/mechanism
pdftex -fmt=pdflatex -interaction=nonstopmode main.tex
bibtex main
pdftex -fmt=pdflatex -interaction=nonstopmode main.tex
pdftex -fmt=pdflatex -interaction=nonstopmode main.tex
```

要求：0 错误、0 未定义引用。
用 PyMuPDF 量行坐标核对无溢出（letter 版心右边界 x1 ≈ 540.3，阈值 541.5），不要靠看图。
commit（push 若网络失败则稍后重试，不强求）。

### 7. 报告用户

完整报告：Q1 截断率表、Q2 判定（R-T1 或 R-T0）、R1200 vs 原 P31 一致性、机制文改了什么。

### 8. 工作日志

在工作区 .workbuddy/memory/2026-11-25.md 追加判读日志（文件不存在则创建）。

## 阶段 A 交接（2026-11-25 14:15 追加；阶段 B 执行者必读，上面步骤 1–8 原文未改）

阶段 A 已完成，commit `22abdcd`（ADR §六 + 读出脚本 + 数据）与 `00c2ee4`（机制文）。

**已做**
- `results/p35_fulltext/p35_race_1200.json` 已入库（2160 行，md5 `64f16bbc…b7f1d`）。
- 读出脚本 **`code/p35_analyze.py`** 已写好并经独立复算核对。阶段 B 直接运行 `PYTHONIOENCODING=utf-8 python code/p35_analyze.py 1200 2400`，
  它会重跑 Q1、一致性检查，并输出 Q2（两种口径 + 按题配对 sign test + R-T1/R-T0 判定 + dup 次要读出）。
  步骤 4 里 Q2 的所有数字都从它取，不要另写一套。
- Q1 结论：M-d 60.9%、M-c 7.0%、M-a 1.7%、M-b 0%（API 原生 `finish_reason == "length"`）；M-d 的 base 步截断 7.4%，88.8% 的截断落在批判调用。
- ADR §六已写入 Q1 与一致性检查；机制文「Residual truncation」段与 Reproducibility 段已按步骤 6 阶段 A 改完并编译（13 页，0 错误，x1 最大 540.3）。

**一致性检查的结果与步骤 6 原文的差别（重要）**
- 步骤 4 末尾写的是「差 < 10pp」。实测 **M-a、M-b、M-d 十二格最大 3.3pp，但 M-c 有两格越线**：
  中位口径 G2 43.3→58.3（+15.0pp）、G3 80.0→70.0（−10.0pp）。已按规则标红、未改 main.tex 正文数字。
- 该红标的性质已查清并记在 ADR §六：**不是行为偏移，是「每题 3 次取中位」口径的抽样放大**。
  行级 top-1 两批最大差 5.0pp；配对 sign test G2 p=0.078、G3 p=0.33；M-d 的 base 步（同一 M-c prompt、独立 540 条）行级 top-1 与两批 M-c 同带。
- **但这不改变一个事实：论文表 2 的 M-c 列对单次重采样敏感。** 用 R1200 替换 P31 重算合并 90 题（exploratory，见 ADR §六），
  终局 M-d − M-c 由 +3.3pp 变为 +11.1pp，G1 的 M-d vs M-c 由 p=.019 变为 p=.071（不再显著），G2 仍显著（p=.006）。
  **R2 否定判决两次都不翻，不重判。** 这是开放问题，处置权在 Lance，阶段 B 不要自行改表 2 或 ADR-0031。
- **【2026-11-25 15:21 更新：此事已由 Lance 定夺，按「维持表 2 + Limitations 披露」执行完毕，本问题关闭。】**
  已在 Limitations 的「Scorer subjectivity」与「Unequal output budget」之间新增一段 **「Three resamples per cell」**
  （M-c 30/37/66 → 33/46/60、M-d 43/63/69 → 44/63/70、终局差 +3 → +10 题、G1 p=.019 → .07，stage-1 领按 suggestive 读）。
  表 2、图 1、§early、附录数字**一律不动**。**阶段 B 不要再碰这一段，也不要改表 2。**
  注意阶段 B 改 Limitations「Unequal output budget」时，它现在紧跟在「Three resamples per cell」之后，别把前一段碰坏。

**阶段 B 必须处理的遗留矛盾**
- 机制文 **Reproducibility 段**现在说重跑批已归档全文，而 **Limitations「Unequal output budget」段**仍写
  "the confirmation batch's full texts were not retained to repeat the census"。阶段 A 按指令没动后者，**两句目前互相矛盾**。
  阶段 B 改 Limitations 段时务必清掉这个矛盾（R-T0 与 R-T1 两个分支都要）。
- Reproducibility 段首句 "the LLM race batches under ADR-0029 through ADR-0032" 未含 ADR-0034，阶段 B 顺手补上（ADR-0033 之后那半句是另一批，别动）。

**R2400 的已知情况**
- 13:02:17 启动，实测约 72 s/行，540 行 → 预计 11-26 00:00 前后完成。若 00:30 仍未完成：按步骤 2「R1200 未完成」同样的逻辑处理进度与 ETA，
  **若预计 1h 内完成则建对应时间的一次性检查任务，否则只报告、不再续建**。
- **2400 token 并未完全消除截断**：中途 51 行里 10 行 `length`（19.6%）。按预注册「不再加码，如实报告」。
  这会影响步骤 6 的措辞——R-T0 分支模板里的 "lifting the cap to 2,400 tokens" 与 R-T1 分支的说明，都要用**实测残余截断率**如实写，
  不要写成「已完全解除」。
- 读 Q2 时注意：sign test 的单位是**题**（同一 puzzle、同一 grade 的 M-d top-1 指示），不是行；行级配对只作敏感性参考。

**环境备忘**
- ssh/scp 单条命令发射，不要用 `&&` 串多条（阶段 A 首次 scp 被前一条命令的非零退出码短路，文件没落盘）。
- 落盘后核对 md5 与服务器一致再读。
- 阶段 A 结束时 `git push` 失败（github.com:443 连不上，本机代理隧道掉线）。阶段 B 先试一次 push，失败就走 `push_via_api.py`，仍失败则如实报告不强求。
