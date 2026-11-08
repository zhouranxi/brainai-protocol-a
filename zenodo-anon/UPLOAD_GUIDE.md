# Zenodo 匿名存档上传指引（TMLR 双盲期）

## 为什么需要匿名账号

Zenodo record 页面永远显示 "Owned by <账号名>"。若用真实账号上传，
审稿人点进 record 即见身份——TMLR 政策不禁止，但匿名实质失效。

## 步骤

1. **注册专用匿名账号** https://zenodo.org/signup
   - 用户名/邮箱不得含真名、单位、常用 ID 任何片段
   - 建议形如 `anon-geo-review-2026@<一次性邮箱域名>`

2. **建 token** https://zenodo.org/account/settings/applications/tokens/new/
   - Name: `anon-supplementary`
   - Scopes: ☑ deposit:write ☑ deposit:actions（两个都勾）

3. **上传**（本目录四个文件，~5.8 MB，本机即可）
   ```bash
   cd zenodo-anon
   export ZENODO_TOKEN='<匿名账号的token>'
   python ../publish_zenodo.py --files-dir . --metadata zenodo_metadata_anon.json
   ```
   脚本停在 draft 状态；先到网页检查标题/描述/license，再点 Publish。
   （注意：publish_zenodo.py 会把 files-dir 下所有文件都上传，包括本
   指南与 metadata json —— 上传前先移走 UPLOAD_GUIDE.md 和
   zenodo_metadata_anon.json，或改用只含三个交付件的子目录。）

4. **发布后**
   - 拿到匿名 DOI（形如 10.5281/zenodo.NNNNNNN）
   - 该 DOI 可写进 OpenReview 投稿的 supplementary 字段或 rebuttal，
     但不要写进论文 PDF（PDF 里的匿名存档句已写明
     "included as supplementary material"，保持不动即可）
   - 录用后在身份版 record 加 `related_identifier` 互链，
     并在 camera-ready 的 Repro 段填身份版 DOI

## 三个交付件

| 文件 | 大小 | 说明 |
|---|---|---|
| paper_causality_anon.pdf | 1.3 MB | causality 论文匿名版（18 页） |
| paper_measurement_anon.pdf | 0.9 MB | measurement 论文匿名版 |
| supplementary.zip | 3.8 MB | 247 成员：代码+数据+结果+图+论文源（匿名化，扫描零命中） |
