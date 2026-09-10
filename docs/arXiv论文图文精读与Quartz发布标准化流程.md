# arXiv 论文深度图文精读与 Quartz/GitHub Pages 发布标准化流程规范 (SOP)

> 本文档基于 **RynnVLA-002 (arXiv: 2511.17502)** 深度图文精读任务的实际落地实践，系统复盘了从论文解析、外链提取、内容精读、Quartz 编译到 GitHub Pages 自动部署全链路中遇到的**核心痛点、工程陷阱与完整解决方案**，形成一套可复用、高可靠的标准化作业流程 (SOP)。

---

## 目录
- [1. 全流程架构概览 (End-to-End Pipeline)](#1-全流程架构概览-end-to-end-pipeline)
- [2. 关键阶段实战问题与避坑解决方案](#2-关键阶段实战问题与避坑解决方案)
  - [阶段一：arXiv 论文 Markdown 与高清图片外链获取](#阶段一arxiv-论文-markdown-与高清图片外链获取)
  - [阶段二：精读文档编写与文本转义规范](#阶段二精读文档编写与文本转义规范)
  - [阶段三：Markdown 表格与学术排版规范 (为什么不用原生 LaTeX Tabular)](#阶段三markdown-表格与学术排版规范-为什么不用原生-latex-tabular)
  - [阶段四：Quartz 静态构建与全要素自动化检查 (Audit Checklist)](#阶段四quartz-静态构建与全要素自动化检查-audit-checklist)
  - [阶段五：Git 提交与 GitHub Pages 自动化发布闭环](#阶段五git-提交与-github-pages-自动化发布闭环)
- [3. 标准化执行脚本与自动化检查工具库](#3-标准化执行脚本与自动化检查工具库)
- [4. 流程全景检查清单 (SOP Checklist)](#4-流程全景检查清单-sop-checklist)

---

## 1. 全流程架构概览 (End-to-End Pipeline)

整个流程分为五个标准阶段：

```text
[阶段一: 论文解析与图床获取]
  arXiv ID -> Hugging Face Papers API / HTML Markdown -> 提取 arXiv 官方图片外链 (200 验证)
       │
[阶段二: 深度解读与文档编撰]
  TL;DR -> 核心痛点 -> 统一架构 -> 实验全解 -> 消融机理 -> 提炼总结 (Python 写入规避转义)
       │
[阶段三: 表格与公式规范化]
  GFM 响应式表格 (严格列数对齐) + 行内 LaTeX/KaTeX 符号 + 块级 LaTeX 公式
       │
[阶段四: Quartz 本地构建与自动化校验]
  npx quartz build -> 自动化脚本审计 (0 Katex Error / 0 未闭合标签 / 0 表格降级)
       │
[阶段五: Git 追踪与 GitHub Pages 发布]
  更新 index.md -> git commit -> push origin main -> GitHub Actions CI 自动部署
```

---

## 2. 关键阶段实战问题与避坑解决方案

### 阶段一：arXiv 论文 Markdown 与高清图片外链获取

#### 遇到的核心问题：
1. **本地下载图片的维护成本极高**：若下载数百 KB 到数十 MB 的插图至 Git 仓库，会迅速膨胀 Git 仓库体积，并容易引发路径相对引用错位。
2. **arXiv HTML 与 Hugging Face Markdown 路径不匹配**：Hugging Face 获取的 Markdown 中，图片常为 `x1.png`、`figures/plot.png` 等相对路径，直接粘贴无法在 Web 端正常展示。

#### 解决方案与规范：
- **直连 arXiv HTML5 官方静态资源池**：
  arXiv 的 HTML 版本（`https://arxiv.org/html/{ARXIV_ID}`）具备固定的静态资产绝对链接：
  - 格式一般为：`https://arxiv.org/html/{ARXIV_ID}v{VERSION}/x1.png`
  - 或具名路径：`https://arxiv.org/html/{ARXIV_ID}v{VERSION}/figures/{filename}.png`
- **外链可达性预检**：在编写文档前，必须通过脚本批量发起 `HEAD` 或 `GET (Range: bytes=0-100)` 请求，确保 HTTP 状态码为 `200` 后再写入文档。

---

### 阶段二：精读文档编写与文本转义规范

#### 遇到的核心问题：
1. **Shell 与编程语言的多重转义崩溃 (Escape Hell)**：
   - 论文解读中不可避免地出现多处 LaTeX 表达式，如 `\alpha`、`\text{...}`、`\mathcal{L}`，以及英文的反引号（如 `'What action should...'`）。
   - 在直接通过 `cat << 'EOF'` 或 JavaScript/Python 字符串模板拼接时，反引号会导致模板字面量意外截断，`\u` 或 `\b`（如 `\begin`）会被误识别为 Unicode 或退格转义符，导致文件生成失败或脚本报错。

#### 解决方案与规范：
- **禁止在包含复杂代码/公式的场景下直接手写嵌套转义**。
- **标准化生成策略**：
  采用独立纯 Python 脚本，以**原始字符串 (Raw String `r"""..."""` 或 base64 编码)** 读取与写入文件，杜绝反引号与斜杠在不同 Shell/引擎之间的解析偏差。

---

### 阶段三：Markdown 表格与学术排版规范 (为什么不用原生 LaTeX Tabular)

#### 遇到的核心问题：
1. **表格分隔符列数不匹配引发“段落降级”**：
   - GFM Markdown 规范中，若表头为 8 列，分隔符行（`| :--- | ... |`）必须严格为 8 个单元格。
   - 若少写或多写了一个单元格，Markdown 解析器将无法将其识别为表格，而是将其**静默降级为普通的段落文本 `<p>`**，导致网页端排版严重崩溃。
2. **能否将表格整体改成原生 LaTeX？**
   - **不能**。Quartz 前端使用的是轻量级 **KaTeX** 渲染引擎，KaTeX 仅支持数学矩阵环境（`\begin{array}`），**完全不支持** LaTeX 的学术表格环境（`\begin{tabular}`、`\begin{table}`、`booktabs` 等）。
   - 若强行使用 `\begin{array}`，数学环境在移动端或窄屏下会以整块公式渲染，无法自动换行或横向滑动，必然造成屏幕右侧大面积破版溢出。

#### 解决方案与规范：
- **统一采用“响应式 Web 表格 + 单元格内嵌 LaTeX”混合模式**：
  - **外层容器**：标准 Markdown 表格，Quartz 会自动为其挂载 `<div class="table-container">`，提供自适应横向平滑滚动条。
  - **单元格内容**：公式符号、希腊字母与指标箭头使用行内 LaTeX，例如：
    - `FVD $\downarrow$`、`PSNR $\uparrow$`
    - `$\pi_0$`、`$\pi_0\text{-FAST}$`
    - 状态标记使用 Unicode 纯文本：`✓` 与 `✗`，避免不必要的公式编译开销。

---

### 阶段四：Quartz 静态构建与全要素自动化检查 (Audit Checklist)

#### 遇到的核心问题：
- 在本地编写完 Markdown 后，往往容易“自我感觉良好”，但直接发布可能存在未渲染公式、坏链、未闭合标签或表格降级等暗病。

#### 解决方案与规范：
发布前执行构建并在 `public/` 目录下运行自动化审查脚本，必须满足 **三零标准**：
1. **0 Katex Error**：HTML 源码中不存在 `class="katex-error"`。
2. **0 Leaked Markdown Table**：HTML 源码中不存在残留未解析的 `| --- |` 纯文本。
3. **0 Broken Image**：文档中声明的 `<img>` 标签数量与 Markdown 插入的原图完全吻合，且外链全部通畅。

---

### 阶段五：Git 提交与 GitHub Pages 自动化发布闭环

#### 遇到的核心问题：
1. 新建文章未在首页索引关联，导致知识孤岛（无法从目录和搜索图谱自然发掘）。
2. GitHub Pages 构建工作流若遇到格式错误可能静默失败。

#### 解决方案与规范：
1. **强制更新 `content/index.md`**：同步增加论文双向链接卡片（`[[paper_id/xxx|标题]]`）、核心标签与 2 行高价值亮点速览。
2. **双重构建保证**：在本地执行 `npx quartz build` 校验无误后再执行 `git push origin main`。
3. **交付物提供**：向用户交付产物时，同步提供在线 GitHub Pages 直达链接与 GitHub 源码链接。

---

## 3. 标准化执行脚本与自动化检查工具库

为实现无人值守质量控制，沉淀以下自动化审计脚本（存放在项目运维规范中）：

### 自动化 HTML 质量审计脚本 (Python)

```python
import os, re

def audit_html(html_file):
    with open(html_file, 'r', encoding='utf-8') as f:
        html = f.read()

    errors = []

    # 1. 检查表格是否降级为文本
    leaked_tables = re.findall(r'\|\s*:?---+', html)
    if leaked_tables:
        errors.append(f'发现 {len(leaked_tables)} 处未正确解析的 Markdown 表格分隔符！')

    # 2. 检查 KaTeX 公式语法错误
    katex_errors = re.findall(r'class="katex-error"[^>]*>(.*?)<', html)
    if katex_errors:
        errors.append(f'发现 {len(katex_errors)} 处 KaTeX 公式语法错误: {katex_errors}')

    # 3. 检查未转义的独立 $$ 公式块
    unrendered_dollar = re.findall(r'\$\$$[^\$]+\$\$$', html)
    if unrendered_dollar:
        errors.append(f'发现 {len(unrendered_dollar)} 处未渲染的 $$ 公式块！')

    # 4. 统计图像数量
    imgs = re.findall(r'<img [^>]*src="([^"]+)"', html)

    print('=== 质量审计结果报告 ===')
    print(f'已解析 HTML 图像总数: {len(imgs)}')
    print(f'表格渲染状态: {"异常" if leaked_tables else "全部通过 (已生成 <table>)"}')
    print(f'公式渲染状态: {"异常" if katex_errors else "全部通过 (无语法错误)"}')

    if errors:
        for err in errors:
            print('❌', err)
        return False
    else:
        print('✅ 页面构建质量 100% 达标！')
        return True
```

---

## 4. 流程全景检查清单 (SOP Checklist)

| 阶段 | 关键动作 | 质检指标 (DoD) |
| :--- | :--- | :--- |
| **1. 准备** | 提取 arXiv ID，抓取 HF API 元数据与官方 HTML 资产 | 确认论文标题、作者机构、代码库及图片 URL 有效 |
| **2. 撰写** | 结构化拆解：TL;DR、痛点、架构、实验、消融、总结 | 覆盖论文全部重点实验表格与关键消融数据 |
| **3. 排版** | Markdown 表格列数严格校验；单元格混编 KaTeX | 表头列数 == 分隔符列数 == 内容数据列数 |
| **4. 构建** | 运行 `npx quartz build`，执行自动化审计脚本 | 0 个 KaTeX 错误，0 个 Markdown 表格降级 |
| **5. 索引** | 更新 `content/index.md`，添加双向链接卡片与标签 | 双向链接激活，图谱关联建立 |
| **6. 发布** | `git add` -> `git commit` -> `git push origin main` | GitHub Actions 部署成功，提供在线与源码链接 |
