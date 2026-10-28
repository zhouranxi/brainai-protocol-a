# Zenodo 发布指引

## 状态：已就绪，只差 access token

发布所需的**所有**文件、metadata、脚本都已准备并验证完毕。唯一缺少的是
Zenodo personal access token —— 这个只能由账号持有人本人创建。

---

## 1. 创建 token（1 分钟）

打开 **https://zenodo.org/account/settings/applications/tokens/new/**

| 字段 | 填什么 |
|---|---|
| Name | `protocol-a-release` |
| Description | `Publication of the Protocol A release` |
| Scopes | ☑ **deposit:write**  ☑ **deposit:actions** |

> ⚠️ **两个 scope 都要勾。** 只有 `deposit:write` 无法发布（publish 动作需要
> `deposit:actions`），只勾 `deposit:actions` 无法上传文件。

生成后 token 只显示一次，立刻复制。

---

## 2. 发布（推荐从服务器传，2.4 GB 直传不走本地带宽）

三个待上传文件已在服务器上：`/mnt/mydisk/xinguan-brainai/release/upload/`

```bash
ssh -i ~/.ssh/id_tonelens_xinguan xinguan@10.10.52.210
cd /mnt/mydisk/xinguan-brainai/release

# 只在当前 shell 会话里设置，不写进任何文件
export ZENODO_TOKEN='粘贴你的token'

# 先只上传、停在草稿状态
python3 publish_zenodo.py \
  --files-dir ./upload \
  --metadata ./zenodo_metadata.json
```

脚本会打印：

```
Created deposition 12345678
  bucket: https://zenodo.org/api/files/...
  draft URL: https://zenodo.org/deposit/12345678
[put ] brainai-protocol-a-code-data-results.tar.gz  (3.5 MB)
  ok in 2s (1.7 MB/s)
[put ] checkpoints_newarch.tar  (322.8 MB)
  ok in 41s (7.9 MB/s)
[put ] checkpoints_resnet18.tar  (2051.6 MB)
  ok in 253s (8.1 MB/s)
```

然后**去 draft URL 人工检查 metadata**，确认无误后有两个选择：

```bash
# 方式 A：网页上点 Publish（推荐——你能亲眼看到最终页面）
# 方式 B：命令行直接发布
python3 publish_zenodo.py --files-dir ./upload --metadata ./zenodo_metadata.json --publish
```

> **注意**：`--publish` 会跳过人工检查直接公开。建议先用不带 `--publish` 的方式跑一次，
> 在 Zenodo 网页上确认标题、作者、描述、license 都对，再点 Publish。
> 脚本支持**断点续传** —— 传到一半断了，重跑同一条命令即可，已传的文件会跳过。

---

## 3. 备用：在本地传（不推荐传 2.4 GB，但小文件可以）

```bash
cd ~/github-repo
export ZENODO_TOKEN='...'
python publish_zenodo.py --files-dir ./zenodo-upload --metadata ./zenodo_metadata.json
```

`zenodo-upload/` 里目前只有 3.5 MB 的代码包。2.4 GB 的 checkpoint 建议走服务器。

---

## 4. 沙箱测试（可选，正式发布前想先验证流程）

沙箱需要**单独注册**并使用**不同的 token**：

```bash
# 1) 到 https://sandbox.zenodo.org 注册账号
# 2) 在 https://sandbox.zenodo.org/account/settings/applications/tokens/new/ 生成 token
export ZENODO_TOKEN='沙箱token'
python3 publish_zenodo.py --sandbox --files-dir ./upload --metadata ./zenodo_metadata.json
```

沙箱 DOI 前缀是 `10.5072`（测试用），正式是 `10.5281`。

---

## 5. 发布后要做的两件事

**① 把 DOI 填进 GitHub 仓库**

```bash
cd ~/github-repo
# 编辑 DATA_AVAILABILITY.md，把 <DOI to be filled at publication time> 换成实际 DOI
# 编辑 zenodo_metadata.json 的 notes 字段（下次发 v2 时用）
git commit -am "Add Zenodo DOI" && git push
```

**② 把 DOI 填进论文**

TMLR 的 camera-ready 版本里，`Reproducibility statement` 一节有
`[repository URL]` 占位符，替换为 Zenodo DOI + GitHub URL。

---

## 上传文件清单

| 文件 | 大小 | 内容 |
|---|---|---|
| `brainai-protocol-a-code-data-results.tar.gz` | 3.5 MB | 代码 + 数据 + 结果 + 图 + 论文源文件（119 个条目） |
| `checkpoints_newarch.tar` | 323 MB | 32 个 checkpoint（16 ShuffleNetV2 + 16 RegNetY） |
| `checkpoints_resnet18.tar` | 2.05 GB | 48 个 ResNet-18 checkpoint |

总计 2.4 GB，3 个文件（远低于 Zenodo 的 50 GB / 100 文件限制）。

## metadata 要点

| 字段 | 值 |
|---|---|
| `upload_type` | `software` |
| `title` | Protocol A: a four-condition control protocol for evaluating geometric markers as out-of-distribution predictors |
| `creators` | Zhou, Ranxi（Hunan University of Chinese Medicine, Department of Computer Science and Technology） |
| `access_right` | `open` |
| `license` | `mit`（已确认是 Zenodo 有效 id） |
| `version` | 1.0.0 |
| `related_identifiers` | GitHub 仓库（isSupplementTo）+ 原论文 arXiv:2603.01879（isDerivedFrom） |

`description` 里已包含：协议四条条件的说明、133 个数字的自验证机制、
checkpoint 的分卷方式与发散模型的处置、以及第三方数据的归属与 md5。
