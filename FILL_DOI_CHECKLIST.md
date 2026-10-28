# DOI 替换清单（Zenodo 发布后执行）

Zenodo 记录发布后，按本清单把真实标识填进两处文件。共 **6 个 `XXXXXXX` 占位符**
分布在 2 个文件里。

---

## 0. 发布

```bash
ssh -i ~/.ssh/id_tonelens_xinguan xinguan@10.10.52.210
cd /mnt/mydisk/xinguan-brainai/release
export ZENODO_TOKEN='...'
python3 publish_zenodo.py --files-dir ./upload --metadata ./zenodo_metadata.json
```

发布成功后脚本会打印：

```
  DOI : 10.5281/zenodo.12345678
  URL : https://zenodo.org/records/12345678
  concept DOI: 10.5281/zenodo.1234567
```

记下这**三个值**：

| 记号 | 含义 | 例 |
|---|---|---|
| `VERSION_DOI` | 本版本的 DOI | `10.5281/zenodo.12345678` |
| `CONCEPT_DOI` | 概念 DOI（跨版本稳定，**引用时优先用它**） | `10.5281/zenodo.1234567` |
| `RECORD_ID` | 记录的数字 ID | `12345678` |

---

## 1. `DATA_AVAILABILITY.md`（4 处）

第 26–29 行，把 `XXXXXXX` 换成实际值：

```markdown
> **Zenodo record:** `10.5281/zenodo.<VERSION_DOI数字部分>` (version DOI)
> **Concept DOI:** `10.5281/zenodo.<CONCEPT_DOI数字部分>`
> **Record page:** `https://zenodo.org/records/<RECORD_ID>`
> **Direct download:** `https://zenodo.org/records/<RECORD_ID>/files/
```

> ⚠️ 注意 version DOI 和 concept DOI **不是同一个数字**。Zenodo 会分配两个：
> version DOI 指向具体这次发布，concept DOI 指向"这一族记录"，后者在未来发 v2 时不变。
> **论文里应引用 concept DOI**（这样 v2 发布后引用依然有效），但也可以同时给出 version DOI。

---

## 2. `paper/main.tex`（Reproducibility statement，2 处，均在第 531 行）

当前写法：

```latex
... archived at \href{https://doi.org/10.5281/zenodo.XXXXXXX}{doi:10.5281/zenodo.XXXXXXX}
(concept DOI \href{https://doi.org/10.5281/zenodo.XXXXXXX}{10.5281/zenodo.XXXXXXX});
the version DOI of this record is 10.5281/zenodo.XXXXXXX.
```

替换为：

```latex
... archived at \href{https://doi.org/10.5281/zenodo.<CONCEPT数字部分>}{doi:10.5281/zenodo.<CONCEPT数字部分>}
(concept DOI \href{https://doi.org/10.5281/zenodo.<CONCEPT数字部分>}{10.5281/zenodo.<CONCEPT数字部分>});
the version DOI of this record is 10.5281/zenodo.<VERSION数字部分>.
```

即：**前两个用 concept DOI，最后一个用 version DOI。**

---

## 3. 重新编译论文

```bash
cd ~/github-repo/paper
export PATH="/c/c/tinytex/bin/windows:$PATH"
export TEXMFHOME=/c/c/tinytex/texmf-var
pdftex -fmt=pdflatex -interaction=nonstopmode main.tex
bibtex main
pdftex -fmt=pdflatex -interaction=nonstopmode main.tex
pdftex -fmt=pdflatex -interaction=nonstopmode main.tex
```

检查：无 error、无 undefined reference、页数与之前一致（18 页）。

> **注意**：Zenodo DOI 是外部链接，hyperref 不会因为它报 undefined reference。但
> 建议编译后用 `pdftotext` 抽一下 Reproducibility statement 那段，确认 DOI 字符串
> 正确渲染、没有被 LaTeX 吞掉字符。

---

## 4. 同步到三处

```bash
cd ~/github-repo
# 论文 PDF 和源文件已改；提交推送
git add -A && git commit -m "Add Zenodo DOI to the reproducibility statement and data availability" && git push
```

同时把更新后的 `paper/main.tex`、`paper/main.pdf` 同步到：

- 服务器 Zenodo 发布包：`/mnt/mydisk/xinguan-brainai/release/brainai-protocol-a/paper/`
- 本地工作副本：`~/WorkBuddy/2026-10-25-21-15-17/brainai-smoke/paper/tmlr/`

---

## 5. 补充：Zenodo 记录本身

Zenodo 记录的 `related_identifiers` 里已包含 GitHub 仓库和原论文 arXiv ID，
无需改动。但**如果论文最终被 TMLR 接收**，建议发一个 Zenodo v2，把 TMLR 的
正式引用信息补进 `notes` 字段——Zenodo 的 version 机制会让 DOI 保持稳定。

---

## 一句话总结

发布 → 拿到 `VERSION_DOI` / `CONCEPT_DOI` / `RECORD_ID` 三个值 →
替换 6 个 `XXXXXXX` → 重编译 → 提交推送。
