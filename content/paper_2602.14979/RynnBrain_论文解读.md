---
title: "RynnBrain: Open Embodied Foundation Models"
date: 2026-10-12
tags:
  - 具身智能
  - 空间智能
  - VLA
  - 视觉-语言-动作模型
  - 具身基础模型
  - 空间时序定位
  - 物理接地推理
  - 强化学习
  - 论文解读
---

# RynnBrain: Open Embodied Foundation Models
> **论文深度精读与技术全解** | 阿里达摩院 (DAMO Academy) · 开源具身时空大脑基础模型 (2B / 8B / 30B-A3B MoE) · 全方位第一人称具身认知 · 跨情节时空精准定位 · 物理接地思维链 (Chain-of-Point, CoP) · 强化学习空间规则奖励建模 · 专用下游模型套件 (RynnBrain-Nav / Plan / VLA) · 21维度高阶具身评测基准 RynnBrain-Bench

---

## 目录
- [1. 论文速览与核心贡献 (TL;DR)](#1-论文速览与核心贡献-tldr)
  - [1.1 论文基本信息与关键元数据](#11-论文基本信息与关键元数据)
  - [1.2 核心研究突破与创新亮点](#12-核心研究突破与创新亮点)
- [2. 研究背景与核心痛点：传统具身大模型的“三大约束”](#2-研究背景与核心痛点传统具身大模型的三大约束)
  - [2.1 通用多模态大模型 (VLM) 缺乏物理世界接地性](#21-通用多模态大模型-vlm-缺乏物理世界接地性)
  - [2.2 传统动作中心化 (Action-Centric) 模型丧失高阶泛化认知](#22-传统动作中心化-action-centric-模型丧失高阶泛化认知)
  - [2.3 既有具身“大脑”的现实缺陷：认知狭窄、依赖单帧、纯文本幻觉](#23-既有具身大脑的现实缺陷认知狭窄依赖单帧纯文本幻觉)
- [3. 核心方法论：物理感知时空预训练体系](#3-核心方法论物理感知时空预训练体系)
  - [3.1 模型基座与三档规模矩阵 (2B, 8B, 30B-A3B MoE)](#31-模型基座与三档规模矩阵-2b-8b-30b-a3b-moe)
  - [3.2 统一时空表示与全模态多任务输入输出空间](#32-统一时空表示与全模态多任务输入输出空间)
  - [3.3 2000万级高保真多模态时空预训练数据飞轮构建](#33-2000万级高保真多模态时空预训练数据飞轮构建)
- [4. 物理接地思维链推理：Chain-of-Point (CoP) 与空间强化学习](#4-物理接地思维链推理chain-of-point-cop-与空间强化学习)
  - [4.1 传统纯文本 CoT 在空间推理中的失效机理](#41-传统纯文本-cot-在空间推理中的失效机理)
  - [4.2 文本与空间坐标交错的 Cold-Start 监督微调 (SFT)](#42-文本与空间坐标交错的-cold-start-监督微调-sft)
  - [4.3 基于强化学习 (PPO/GRPO) 的可验证空间规则奖励建模](#43-基于强化学习-ppogrpo-的可验证空间规则奖励建模)
- [5. 下游专用具身套件：导航、规划与动作控制](#5-下游专用具身套件导航规划与动作控制)
  - [5.1 RynnBrain-Nav：流式连续视觉语言导航 (VLN-CE)](#51-rynnbrain-nav流式连续视觉语言导航-vln-ce)
  - [5.2 RynnBrain-Plan：细粒度多轮交互物理操作规划器](#52-rynnbrain-plan细粒度多轮交互物理操作规划器)
  - [5.3 RynnBrain-VLA：基于流匹配/扩散的高精度底层动作执行器](#53-rynnbrain-vla基于流匹配扩散的高精度底层动作执行器)
  - [5.4 端到端云-边分层具身闭环协作部署](#54-端到端云-边分层具身闭环协作部署)
- [6. 实验结果与深度定量分析](#6-实验结果与深度定量分析)
  - [6.1 RynnBrain-Bench：首个全景细粒度时空具身基准](#61-rynnbrain-bench首个全景细粒度时空具身基准)
  - [6.2 8B 规模以下具身基础模型全面对决 (Table 3)](#62-8b-规模以下具身基础模型全面对决-table-3)
  - [6.3 30B+ 规模与前沿专有大模型 (GPT-5.2, Gemini 3 Pro) 极限对决 (Table 4)](#63-30b-规模与前沿专有大模型-gpt-52-gemini-3-pro-极限对决-table-4)
  - [6.4 空间思维链推理定量评测：超越前沿思考模型 (Table 5)](#64-空间思维链推理定量评测超越前沿思考模型-table-5)
  - [6.5 真实环境与虚拟基准连续导航极限评测 (Table 6)](#65-真实环境与虚拟基准连续导航极限评测-table-6)
  - [6.6 真实世界多物体操作与多轮规划消融 (Table 7 & Table 8)](#66-真实世界多物体操作与多轮规划消融-table-7--table-8)
- [7. 局限性与未来探索方向](#7-局限性与未来探索方向)
- [8. 核心要点总结 (Key Takeaways)](#8-核心要点总结-key-takeaways)

---

## 1. 论文速览与核心贡献 (TL;DR)

### 1.1 论文基本信息与关键元数据
- **论文标题**：RynnBrain: Open Embodied Foundation Models
- **作者团队**：Jiayan Guo, Bohan Hou, Sicong Leng, Kehan Li, Xin Li, Jiangpin Liu, Yunxuan Mao, Zhikai Wang, Yuqian Yuan, Minghao Zhu, Xiao Lin, Yang Bai, Qian Jiang, Yaxi Zhao, Minghua Zeng, Junlong Gao, Yuming Jiang, Jun Cen, Siteng Huang, Liuyi Wang, Wenqiao Zhang, Chengju Liu, Jianfei Yang, Shijian Lu, Deli Zhao
- **所属机构**：阿里巴巴达摩院 (DAMO Academy, Alibaba Group)
- **发布时间**：2026年2月 (arXiv: 2602.14979v1)
- **开源协议**：全系列模型权重与核心算法均以 **Apache 2.0** 协议完全开源
- **论文链接**：[arXiv:2602.14979](https://arxiv.org/abs/2602.14979)

![RynnBrain 系统总览图](https://arxiv.org/html/2602.14979v1/intro.png)
*图 1：RynnBrain 具身基础模型总体架构概览。统一输入多视角图像、长时序视频与三维空间坐标；统一输出自然语言、边界框、交互热点点位 (Affordance)、操作区域 (Area) 与操作轨迹 (Trajectory)，全面支撑第一人称具身认知、时空定位、物理推理与物理感知规划四大核心支柱。*

### 1.2 核心研究突破与创新亮点
1. **统一的时空具身“大脑”基础模型 (Open Embodied Brain)**：不同于传统仅能看单帧静态图像做文本问答的通用 VLM，或牺牲高层语义认知的低层动作模型，RynnBrain 基于 Qwen3-VL 打造，覆盖 **2B、8B、30B-A3B (MoE 架构，激活参数仅 3B)** 三档模型矩阵，在保持强大通用多模态理解能力的同时，全面注入物理世界时空接地先验。
2. **首创物理接地思维链 (Chain-of-Point Reasoning, CoP)**：彻底颠覆传统大模型纯文字推理 (Textual CoT) 容易“凭空捏造”的弊端，提出**思维链中文字推理与空间定位图元交错生成**机制。引入结合可验证物理几何规则的强化学习 (PPO/GRPO) 奖励机制，在接触点位、区域与轨迹预测上全面超越 GPT-5.2 与 Gemini 3 Pro。
3. **近 2000 万级 (19.89M) 全维度时空具身预训练数据飞轮构建**：融合通用 MLLM (4.8M)、多维第一人称认知 (7.67M，涵盖 3D 点云与重力对齐坐标系重构的空间几何数据、细粒度视频 OCR、目标理解与计数)、时空定位 (7.26M，涵盖边界框、区域点集、Affordance 关键点、2D 轨迹及 4 顶点定向抓取姿态) 以及物理操作规划数据 (0.16M)。
4. **全套下游落地专精派生套件 (Post-Trained Suite)**：派生推出 **RynnBrain-CoP** (空间推理专用)、**RynnBrain-Nav** (VLN 连续导航 SOTA，R2R-CE 成功率 58.6%，Oracle 成功率高达 71.6%)、**RynnBrain-Plan** (分层细粒度规划，结合人类 UMI 接口攻克长程复杂家务整理) 以及 **RynnBrain-VLA** (直接回归底层机械臂连续控制指令，物体识别拾取成功率达 80%)，构建了完整的具身智能全栈技术底座。

---

## 2. 研究背景与核心痛点：传统具身大模型的“三大约束”

### 2.1 通用多模态大模型 (VLM) 缺乏物理世界接地性
以 GPT-4o、Gemini 为代表的通用多模态大语言模型，展现出了令人惊叹的语义理解、常识归纳与开集对话能力。然而，直接将通用 VLM 迁移至机器人具身物理场景时，暴露出严重的**物理世界脱节**：
- **无尺度度量感知 (Lack of Metric Sense)**：无法可靠感知相机的旋转、场景绝对尺度、物体之间的厘米级欧几里得距离与物理高度。
- **缺乏时空一致性 (Temporal Inconsistency)**：机器人移动过程中的第一人称视频视角变化剧烈，通用 VLM 难以维护多视角或动态情节下的持久空间记忆 (Episodic Memory)。

### 2.2 传统动作中心化 (Action-Centric) 模型丧失高阶泛化认知
与通用 VLM 走向另一极端的是以 OpenVLA、$\pi_0$ 等为代表的具身端到端 VLA 模型。这些模型直接在机械臂遥操或仿真轨迹上进行大规模动作模仿学习 (Imitation Learning)。虽然它们能够执行高频平滑控制，但存在致命软肋：
- **高阶语义抽象能力退化**：为了拟合特定的末端执行器动作空间，往往牺牲了语言模型的深层推理和开集常识能力。
- **任务适应极其脆弱**：一旦遇到未见过的指令、杂乱背景或需要多步分解的长程因果推理任务，单纯的动作策略网络极易陷入停滞或执行死循环。

### 2.3 既有具身“大脑”的现实缺陷：认知狭窄、依赖单帧、纯文本幻觉
近年来，学术界尝试构建介于通用 VLM 与底层控制之间的具身“大脑”模型 (如 RoboBrain 2.0、Pelican-VL、MiMo-Embodied 等)，但它们普遍面临三项关键瓶颈：
1. **第一人称认知能力狭窄 (Narrow Egocentric Cognition)**：大多局限于简单的桌面静态物体描述，完全忽视了真实移动操作中不可或缺的长时序视频理解、细粒度动态 OCR、空间遮挡判断与跨帧计数。
2. **时空定位依赖单帧静态输入 (Static-Only Spatial Grounding)**：仅支持在单一截图上输出 2D Bounding Box，缺少跨越机器人移动全生命周期的连续轨迹预测、可交互功能点位 (Affordance) 与可通行/可放置区域 (Area) 的全局建模能力。
3. **高阶推理的“纯文本幻觉” (Purely Textual Hallucination)**：思考过程仅以自然语言展开（例如“我应该走向桌子然后拿杯子”），这种未受几何与物理约束的文本推理极易出现脱离实际物理限制的幻觉（如命令机械臂穿墙或抓取视野外不可触达的物体）。

---

## 3. 核心方法论：物理感知时空预训练体系

![RynnBrain 架构图](https://arxiv.org/html/2602.14979v1/rynnbrain_framework_v3.png)
*图 2：RynnBrain 统一模型架构细节。支持全景单视角、多视角图像序列及连续视频流输入，配合语言提示词，通过统一的多模态解码器 (Dense 或 MoE)，联合自回归生成结构化文本、边界框、点位集合、连续轨迹以及带有定向角度的抓取矩形。*

### 3.1 模型基座与三档规模矩阵 (2B, 8B, 30B-A3B MoE)
为兼顾边缘端低算力实时推理与云端超大规模复杂空间规划，RynnBrain 基于新一代多模态基座 **Qwen3-VL** 进行全参数深度重塑预训练，构建了三套规格化基座：
- **RynnBrain-2B**：专为边缘计算芯片（如车载计算单元、机器人板载计算棒）设计，极致轻量，具备高速推理与基础空间定位能力。
- **RynnBrain-8B**：高性价比主力工作模型，兼具强大的物理空间推理、长视频解析与多步任务规划能力，单卡 RTX 3090/4090 即可高效部署。
- **RynnBrain-30B-A3B**：采用 **Mixture-of-Experts (MoE)** 稀疏架构。总参数量为 30B，但每个 Token 仅动态激活 3B 参数，既拥有 30B 级别的世界知识容量与深层因果逻辑，又保持了接近 3B 密集模型的超高推理吞吐量。

| 参数项 | RynnBrain-2B | RynnBrain-8B | RynnBrain-30B-A3B |
| :--- | :---: | :---: | :---: |
| **基础骨干模型 (Base Model)** | Qwen3-VL-2B-Instruct | Qwen3-VL-8B-Instruct | Qwen3-VL-30B-A3B-Instruct |
| **优化器 (Optimizer)** | AdamW | AdamW | AdamW |
| **语言与投影层学习率 ($lr_{\text{LLM}}$)** | $5 \times 10^{-6}$ | $2 \times 10^{-6}$ | $2 \times 10^{-6}$ |
| **视觉编码器学习率 ($lr_{\text{Vision}}$)** | $1 \times 10^{-6}$ | $2 \times 10^{-6}$ | $2 \times 10^{-6}$ |
| **全局批量大小 (Global Batch Size)** | 512 | 1024 | 1024 |
| **预热比例 (Warmup Ratio)** | 0.03 (3%) | 0.03 (3%) | 0.03 (3%) |
| **最大上下文长度 (Max Context Length)** | 16,384 tokens | 16,384 tokens | 16,384 tokens |

### 3.2 统一时空表示与全模态多任务输入输出空间
为了让同一套模型能够统一处理从感知、提问到控制规划的全部任务，RynnBrain 将物理几何空间离散化并映射到自回归词表空间中，建立了统一的时空多模态接口：

1. **多模态全景感知输入**：
   输入形式形式化为视频/多视角帧序列 $\mathbf{V} = \{I_t\}_{t=1}^T$（其中静态图像 $T=1$）与用户自然语言任务指令 $Q$。针对长程视频，采用 2 FPS 均匀下采样，单次输入支持高达 2048 帧视觉输入。
2. **统一物理几何输出图元**：
   - **目标边界框 (Bounding Box)**：$\mathcal{B} = (x_0, y_0, x_1, y_1)$，坐标归一化到 $[0, 1000]$ 整数离散空间，并显式绑定关键帧索引 $t$：

$$
\langle\text{object}\rangle \; \langle\text{frame } t\rangle\text{: } (x_0, y_0, x_1, y_1) \; \langle/\text{object}\rangle
$$

   - **可通行/可放置区域 (Area Points)**：采用点集表示非刚性或平铺表面 $\mathcal{P} = \{(x_i, y_i)\}_{i=1}^n$：

$$
\langle\text{area}\rangle \; \langle\text{frame } t\rangle\text{: } (x_1, y_1), \dots, (x_n, y_n) \; \langle/\text{area}\rangle
$$

   - **可交互功能点 (Affordance Point)**：单个关键交互热点 $p = (x, y)$（例如把手中心、电源按钮点位）：

$$
\langle\text{affordance}\rangle \; \langle\text{frame } t\rangle\text{: } (x, y) \; \langle/\text{affordance}\rangle
$$

   - **空间操作轨迹 (Trajectory)**：时间有序的 2D 引导轨迹 $\mathcal{T} = \{(x_i, y_i)\}_{i=1}^m$（最多 $m=10$ 个有序路标点），起始于关键帧 $t_s$：

$$
\langle\text{trajectory}\rangle \; \langle\text{start } t_s\rangle\text{: } (x_1, y_1), \dots, (x_m, y_m) \; \langle/\text{trajectory}\rangle
$$

   - **机器人四顶点定向抓取矩形 (Grasp Pose)**：$\mathcal{G} = \{(x_i, y_i)\}_{i=1}^4$，将以中心、长宽和旋转角 $(c_x, c_y, w, h, \theta)$ 描述的夹爪位姿转换为 4 个有序角点，显式捕捉末端抓取方向。

### 3.3 2000万级高保真多模态时空预训练数据飞轮构建
RynnBrain 采集并清洗了共计 **19.89M (近 2000 万)** 样本的超大规模多源预训练数据集，覆盖通用认知、多维具身理解、时空定位与物理规划全维度：

| 大类类别 (Category) | 子任务分支 (Sub-Task) | 主要数据源 (Data Sources) | 样本量 (M) |
| :--- | :--- | :--- | :---: |
| **通用多模态 (General MLLM)** | 通用跨模态图文与视频理解 | LLaVA-OV-SI, LLaVA-Video, ShareGPT-4o-video, VideoGPT+, FineVideo, CinePile, ActivityNet, YouCook2, LLaVA-SFT, VideoLLaMA 3 | **4.80** |
| **具身认知 (Cognition)** | 细粒度物体理解 (Object Understanding) | RynnBrain-Object (自采室内第一人称+SAM2跟踪标注), RefCOCO, Google Refexp, Osprey-724K, DAM, VideoRefer-700k | 1.10 |
| | 空间几何理解 (Spatial Understanding) | Sensenova-SI-800K, VSI-590k, VLM-3R, RynnBrain-Spatial (MASt3R-SLAM 点云重构 + RANSAC 重力对齐) | 2.50 |
| | 复杂场景计数 (Counting) | Molmo2 counting, RynnBrain-Counting (第一人称高精度质检) | 0.30 |
| | 细粒度视频 OCR (OCR) | Ego4D, Charades-Ego, EPIC-KITCHENS 提取视频段 + GoMatching 识别 + GPT-5.2 第一人称问答与结构化坐标 | 1.00 |
| | 第一人称任务理解 (Egocentric Task) | EgoRe-5M, EgoTaskQA, Env-QA, QAEgo4D, RoboVQA, Robo2VLM, ShareRobot (过滤 <3s 片段) | 2.77 |
| **时空定位 (Localization)** | 目标物体检测定位 (Object Location) | ADE20K, COCOStuff, Mapillary, PACO-LVIS, PASCAL-Part, VG, RoboAfford-Object, RynnBrain-Grounding | 1.20 |
| | 空间区域定位 (Area Location) | RefSpatial, RoboAfford-Area, Molmo2, RynnBrain-Area (自采 6K 家居视频+点位标注) | 3.37 |
| | 可交互点定位 (Affordance Location) | RynnBrain-Affordance, RoboAfford-Affordance (手柄、按键交互热点) | 1.13 |
| | 2D 轨迹规划定位 (Trajectory Prediction) | RynnBrain-Trajectory (6K 跨帧长视频轨迹 + 507K 静态图像), FSD | 0.56 |
| | 定向抓取位姿估计 (Grasp Pose) | Grasp-Anything (945K 真实图像提取 1.3M 样本，四角点定向矩形) | 1.00 |
| **物理规划 (Planning)** | 原子动作与空间接地规划 (Manipulation) | AgibotWorld Alpha, Open X-Embodiment, RynnBrain-Planning (带框/点/区域注释) | **0.16** |
| **全量总计 (Total)** | **覆盖四大支柱全维度时空具身能力** | **跨源整合与自研管线高保真清洗** | **19.89** |

---

## 4. 物理接地思维链推理：Chain-of-Point (CoP) 与空间强化学习

![思维链推理示例](https://arxiv.org/html/2602.14979v1/example_thinking.png)
*图 3：RynnBrain-CoP 物理接地思维链推理可视化样例。模型在生成文字思考推演的同时，自主在对应的视频帧中输出目标物体框与关键区域坐标，实现纯逻辑推导与物理感知证据的紧密耦合。*

### 4.1 传统纯文本 CoT 在空间推理中的失效机理
当前主流思考模型 (Thinking Models，如 OpenAI o1/o3、DeepSeek-R1) 依赖 <think>...</think> 形式的纯自然语言链条。当应用于需要高精度几何定位的具身任务时，纯文本 CoT 暴露出明显缺陷：
- 语言可以描述“应该抓住杯子把手”，但无法量化把手在当前视场中的具体欧氏坐标；
- 中间推理步骤缺乏物理可验证证据，导致大模型在多步自回归中逐渐累积定位误差，发生严重的“幻觉漂移”。

### 4.2 文本与空间坐标交错的 Cold-Start 监督微调 (SFT)
RynnBrain 提出 **Chain-of-Point (CoP)** 机制，在思考链内部直接**交织物理空间定位图元**。
1. **数据构建管线**：
   - 首先利用超大模型 Qwen3-VL-235B 针对任务视频生成基础推理链条，并在文本中将候选实体使用方括号标记（如 [白色花纹壁纸]）；
   - 使用实体分类模型判断该实体属于“物体”还是“区域”；
   - 由人类标注专家与几何检测模型协同，在时序帧中选出最清晰关键帧，插入精确的点集或边界框，形成交错式推理结构：
     ```text
     <think>
     为了完成将水杯放入微调架的任务，首先需要在场景中定位杯架的目标置物层。
     <area> <frame 12>: (421, 650), (455, 680), (510, 660) </area>
     接着，观察杯子当前的摆放姿态与把手朝向，确定最利于平稳抓取的接触点。
     <affordance> <frame 18>: (312, 458) </affordance>
     规划从当前末端初始位置到达抓取点的引导轨迹。
     </think>
     最终动作规划：...
     ```
2. **SFT 训练细节**：
   - 以预训练好的 RynnBrain 为起点，采用 AdamW 优化器，余弦衰减调度；
   - 语言层与投影层峰值学习率 $1 \times 10^{-5}$，视觉编码器 $2 \times 10^{-6}$，预热 3%；
   - 帧率 2 FPS（最高 2048 帧），最大序列长度 16,384 tokens，批量大小 128，采用 DeepSpeed ZeRO-1 进行显存优化。

### 4.3 基于强化学习 (PPO/GRPO) 的可验证空间规则奖励建模
仅靠 SFT 难以穷尽复杂的几何长尾情况。RynnBrain 进一步引入强化学习 (RL)，针对物理坐标设计了**确定性、可验证的规则几何奖励函数 (Rule-Based Spatial Reward)**，彻底告别了传统 RLHF 中难以收敛的模糊偏好模型：

1. **轨迹奖励函数 (Trajectory Reward, $R_{\text{traj}}$)**：
   计算预测轨迹点序列 $\hat{\mathcal{T}} = \{\hat{p}_i\}_{i=1}^m$ 与真实标注轨迹 $\mathcal{T}^* = \{p_i^*\}_{i=1}^m$ 的平均归一化欧氏距离 (MSE) 及倒角距离 (Chamfer Distance)：

$$
R_{\text{traj}} = \exp\left( - \frac{1}{m} \sum_{i=1}^m \frac{\|\hat{p}_i - p_i^*\|_2^2}{2\sigma_{\text{traj}}^2} \right)
$$

2. **交互热点奖励函数 (Affordance Reward, $R_{\text{aff}}$)**：
   针对接触关键点 $\hat{p}$ 与目标真实点 $p^*$，引入带有截断阈值 $\tau$ 的平滑高斯核奖励：

$$
R_{\text{aff}} = \begin{cases} \exp\left( - \frac{\|\hat{p} - p^*\|_2^2}{2\sigma_{\text{aff}}^2} \right), & \text{if } \|\hat{p} - p^*\|_2 \le \tau \\ 0, & \text{otherwise} \end{cases}
$$

3. **区域定位奖励函数 (Area Reward, $R_{\text{area}}$)**：
   度量预测点集 $\hat{\mathcal{P}}$ 落在真实目标多边形区域 $\Omega^*$ 内的覆盖率与双向倒角距离：

$$
R_{\text{area}} = \alpha \cdot \frac{|\hat{\mathcal{P}} \cap \Omega^*|}{|\hat{\mathcal{P}}|} + (1-\alpha) \cdot \exp\left( - d_{\text{Chamfer}}(\hat{\mathcal{P}}, \Omega^*) \right)
$$

4. **格式合规性奖励 ($R_{\text{format}}$)**：
   若模型输出了合法的 XML 标签配对、合法的坐标数字区间 $[0, 1000]$ 且帧索引在有效视频范围内，给予固定奖励 $+1$，否则给予惩罚 $-1$。

通过这种纯客观几何规则奖励驱动，RynnBrain-CoP 在思考阶段自发形成了“观察视场 $\rightarrow$ 框选目标 $\rightarrow$ 计算落点几何中心 $\rightarrow$ 生成最终坐标”的物理接地推演能力。

---

## 5. 下游专用具身套件：导航、规划与动作控制

针对真实机器人落地的不同环节，RynnBrain 衍生了三大下游专用微调模型，覆盖从宏观探索、中层任务分解到底层连续控制的全链路：

### 5.1 RynnBrain-Nav：流式连续视觉语言导航 (VLN-CE)
针对连续环境视觉-语言导航任务 (VLN-CE)，将预训练 RynnBrain 在 Matterport3D 模拟器（包含 R2R 与 ScaleVLN 轨迹数据共超 30 万条）上进行多轮对话式流式微调。
- **动作离散化**：$\mathcal{A} = \{\uparrow (\text{前进 30cm}), \leftarrow (\text{左转 } 15^\circ), \rightarrow (\text{右转 } 15^\circ), \text{STOP}\}$。
- **数据增强飞轮**：采用多轮 DAgger (Dataset Aggregation) 收集智能体偏航后的自修复数据，大幅提升长程避障与死胡同解脱能力。

### 5.2 RynnBrain-Plan：细粒度多轮交互物理操作规划器
作为云端高级决策大脑，RynnBrain-Plan 负责将高度抽象的模糊人类指令（例如“帮我收拾乱七八糟的餐桌并分类厨具”）分解为严格有序的原子级子任务。
- **混合图元输出**：每个步骤输出均携带具体操作物体的包围框 $\mathcal{B}$、放置目标多边形中心点 $\mathcal{P}$ 以及抓取把手点位 $p$。
- **支持多轮环境反馈 (Multi-Turn Dialogue)**：在执行过程中，若环境状态由于外力干扰或执行失误发生意外改变，RynnBrain-Plan 能根据最新视觉感知动态修正后续规划步骤。

### 5.3 RynnBrain-VLA：基于流匹配/扩散的高精度底层动作执行器

![RynnBrain-VLA 架构图](https://arxiv.org/html/2602.14979v1/vla_model.png)
*图 4：RynnBrain-VLA 统一视觉-语言-动作模型架构。以预训练 RynnBrain 视觉特征为条件输入，通过轻量级 Flow Matching / Diffusion 动作生成头，以 10-50Hz 高频直接回归机械臂末端 6-DoF 位姿轨迹与夹爪开合状态。*

在机械臂高频控制环节，利用 RynnBrain 作为强力冻结/微调视觉骨干，后接动作扩散/流匹配头 (Flow Matching Action Decoder)：
- 接收初始观测图像与当前机械臂末端状态 $s_t$；
- 采用连续流匹配模型在潜在动作空间中进行去噪采样，预测未来连续时间步的末端速度与夹爪二值开合度；
- 摆脱了传统 VLA 模型对预训练动作数据集的极端过拟合，具备卓越的跨物体与杂乱背景抗干扰能力。

### 5.4 端到端云-边分层具身闭环协作部署
RynnBrain 架构天然支持**分层协同 (Hierarchical Collaboration)**：
- **云端/中心计算节点 (The Brain)**：运行 RynnBrain-Plan-30B-A3B，以 1~2Hz 的频率监控全局场景，负责语义理解、因果推理并输出带有空间点位的结构化规划；
- **边缘/本体计算节点 (The Cerebellum)**：运行 RynnBrain-VLA 或低级控制器，接收高层下发的局部目标框与接触点，以 30Hz 高频执行实时的避障抓取与精细闭环操作。

---

## 6. 实验结果与深度定量分析

### 6.1 RynnBrain-Bench：首个全景细粒度时空具身基准

![RynnBrain-Bench 维度图](https://arxiv.org/html/2602.14979v1/rynnbrain_bench.png)
*图 5：RynnBrain-Bench 评测维度体系。包含两大赛道（具身认知与时空定位），共涵盖 21 项细粒度核心能力指标，由 3,616 段精细视频片段、577,998 帧图像及 12,000 道多轮交叉校验问题组成。*

为彻底解决现有基准仅测静态单图、指标与真实具身严重脱节的问题，团队推出了 **RynnBrain-Bench**，涵盖 4 大基础支柱与 21 个专业子能力：
1. **物体认知 (Object Cognition)**：类别判定、物体颜色、几何形状、物理功能属性、空间相对拓扑等；
2. **空间认知 (Spatial Cognition)**：相机自身平移与旋转角度估计、物体绝对公制距离、相对朝向方位、物体三维尺寸；
3. **空间定位接地 (Grounding)**：文本指代物体框定位 (Object Grounding)、放置/工作区域多边形定位 (Area Location)；
4. **物理交互指引 (Pointing)**：可交互把手/按钮热点定位 (Affordance Pointing)、2D 轨迹引导生成 (Trajectory Generation)。

### 6.2 8B 规模以下具身基础模型全面对决 (Table 3)
在 8B 及以下参数规模的主流开源模型评测中，RynnBrain 与 MiMo-Embodied-7B、RoboBrain 2.0-7B、Pelican-VL-7B、Cosmos-reason2-8B 及基座 Qwen3-VL-8B 进行了全面对比（带 * 号为官方复现结果）：

| 评测大类 | 评测基准数据集 | RynnBrain-2B | RynnBrain-8B | MiMo-Embodied-7B | RoboBrain 2.0-7B | Pelican-VL-7B | Cosmos-reason2-8B | Qwen3-VL-8B |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **具身认知** | VSI-Bench | 70.5 | **71.0** | 48.5 | 36.1 | 52.8 | 53.7* | 60.3 |
| (Embodied Cognition) | MMSI | 34.1 | **39.6** | 30.2* | 24.8* | 26.2* | 31.3* | 29.6 |
| | ERQA | 42.3 | **46.8** | **46.8** | 36.5* | 39.8* | 46.0* | 44.8 |
| | RoboSpatial | 65.7 | **73.1** | 61.8 | 54.2 | 57.5 | 59.0* | 58.2 |
| | EgoTaskQA | **73.9** | 72.5 | 58.7* | 51.1* | 50.0* | 55.7* | 57.8 |
| | EgoTextVQA (Indoor) | 27.7 / 2.08 | 31.6 / 2.28 | 28.7 / 2.17* | 22.0 / 1.79* | 30.3 / 2.24* | 26.5 / 1.96* | **38.9 / 2.64*** |
| | Open-X VQA | 71.0 | **74.0** | 41.5* | 44.6 | 44.1* | 55.0* | 59.8 |
| | QAEgo4D | 43.9 | 43.9 | 39.0* | 39.7* | 26.1* | **46.9*** | 44.0 |
| | MindCube | 50.1 | **56.6** | 43.1* | 38.9 | 33.7* | 43.9* | 36.0 |
| | RynnBrain-Object | 70.7 | **71.2** | 39.0 | 24.7 | 30.8 | 37.2 | 41.8 |
| | RynnBrain-Spatial | 57.2 | **59.9** | 28.3 | 13.5 | 20.5 | 31.4 | 35.0 |
| **时空定位** | RefSpatial-Bench | 52.7 | **59.2** | 48.0 | 32.5 | 22.3 | 33.1* | 53.4 |
| (Embodied Location) | ShareRobot-Affordance | 43.3 | **44.7** | 35.8* | 28.1 | 11.3 | 37.1* | 37.0 |
| | ShareRobot-Trajectory ($\downarrow$) | **0.34** | 0.35 | 0.41* | 0.55 | 0.42* | 0.36* | 0.37 |
| | Cornell-Grasp | 20.9 | **26.6** | 0.2* | 0.0* | 0.0* | 18.1* | 21.2* |
| | VMRD-Grasp | 13.0 | **14.1** | 2.8* | 0.5* | 0.0* | 13.7* | 7.1* |
| | RynnBrain-Grounding | 79.1 | **81.6** | 49.8 | 18.6 | 3.5 | 60.0 | 62.8 |
| | RynnBrain-Area | 54.6 | **56.2** | 49.4 | 38.0 | 46.5 | 37.6 | 30.0 |
| | RynnBrain-Affordance | 89.4 | **90.4** | 84.4 | 73.5 | 81.4 | 83.9 | 82.9 |
| | RynnBrain-Trajectory | **66.6** | 64.5 | 61.3 | 57.6 | 59.2 | 64.0 | 63.4 |
| **通用视觉理解** | AI2D | 79.4 | **86.3** | 84.2 | 70.3* | 83.8* | 83.0* | 85.7 |
| (General Understanding)| ChartQA | 78.2 | 86.5 | 85.2* | 82.4* | 87.5* | 84.3* | **89.6** |
| | DocVQA (Val) | 93.0 | 96.2 | 94.9* | 93.1* | 94.5* | 95.0* | **96.4** |
| | MVBench | 67.3 | **69.5** | 57.9* | 50.6* | 67.7 | 67.0* | 68.7 |
| | RealWorldQA | 60.4 | 67.3 | 66.1* | 51.2* | 67.1* | 69.3* | **71.5** |
| | InfoVQA (Test) | 71.2 | **83.4** | 72.0* | 77.6* | 81.1* | 78.3* | 83.1 |
| | EgoSchema | 64.0 | **69.7** | 58.2* | 54.2* | 73.3 | 63.5* | **69.7** |
| | VideoMME (w/o sub) | 61.4 | 70.7 | 65.0* | 52.3* | 63.3 | **71.9** | 71.4 |

> **关键技术洞察**：
> 1. **2B 模型即可跨级别抗衡竞品 7B/8B**：RynnBrain-2B 在 VSI-Bench (70.5)、RoboSpatial (65.7)、Open-X VQA (71.0) 以及 RynnBrain-Grounding (79.1) 上不仅远超 7B 规模的 RoboBrain 2.0 与 Pelican-VL，甚至超越了 8B 级的基座 Qwen3-VL。
> 2. **空间理解能力质的飞跃**：在专门评估 3D 空间关系的 RynnBrain-Spatial 上，RynnBrain-8B 达到 59.9，比此前领先的通用多模态模型提升近一倍 (35.0 $\rightarrow$ 59.9)，证明了 MASt3R-SLAM 点云对齐数据的巨大价值。
> 3. **保留了强悍的通用视觉常识**：在引入近 2000 万具身数据微调后，RynnBrain-8B 在 DocVQA (96.2)、AI2D (86.3) 和 VideoMME (70.7) 等通用 Benchmark 上完全未出现灾难性遗忘。

### 6.3 30B+ 规模与前沿专有大模型 (GPT-5.2, Gemini 3 Pro) 极限对决 (Table 4)
在 30B 及以上超大规模量级中，RynnBrain-30B-A3B 与工业界顶尖专有大模型及开源大模型展开对决：

| 评测大类 | 评测基准数据集 | RynnBrain-30B (A3B) | RoboBrain 2.0-32B | Pelican-VL-72B | GPT-5.2 | Gemini 3 Pro | Claude Sonnet 4.5 | Qwen3-VL-30B (A3B) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **具身认知** | VSI-Bench | **74.5** | 42.7 | 57.3 | 46.6* | 48.8* | 42.5* | 65.8 |
| (Embodied Cognition) | MMSI | 39.5 | 28.5* | 30.7* | 38.2* | **49.2** | 28.9* | 21.1 |
| | ERQA | 46.3 | 46.0 | 43.0 | 45.3* | **70.5** | 60.0 | 43.0 |
| | RoboSpatial | 70.0 | **72.4** | 55.4 | 54.7* | 56.0* | 40.9* | 55.4 |
| | EgoTaskQA | **78.9** | 59.9* | 64.8* | 59.6* | 68.4* | 50.9* | 64.2* |
| | EgoTextVQA (Indoor) | 34.6 / 2.39 | 30.5 / 2.24* | 37.8 / 2.59* | **49.6 / 3.02*** | 45.5 / 2.87* | 36.6 / 2.55* | 41.3 / 2.74* |
| | Open-X VQA | **83.4** | 28.6* | 48.0* | 43.6* | 56.0* | 41.9* | 76.8* |
| | QAEgo4D | **47.3** | 40.3* | 24.6* | 46.8* | 42.1* | 35.0* | **47.3*** |
| | MindCube | 63.4 | 29.2 | 32.5* | 61.7 | **70.8** | 58.3 | 39.0* |
| | RynnBrain-Object | **73.3** | 26.2 | 42.2 | 53.1 | 44.6 | 25.1 | 42.6 |
| | RynnBrain-Spatial | **59.3** | 11.6 | 32.2 | 33.7 | 29.0 | 34.2 | 30.7 |
| **时空定位** | RefSpatial-Bench | 59.2 | 54.0 | 49.5 | 26.4* | **65.5** | 15.1 | 53.1 |
| (Embodied Location) | ShareRobot-Affordance | 43.2 | 35.3 | 10.4* | 17.5* | 26.9* | 13.9 | **47.2*** |
| | ShareRobot-Trajectory ($\downarrow$) | 0.31 | **0.24** | 0.36* | 0.35* | 0.29* | 0.57 | 0.36* |
| | Cornell-Grasp | **33.6** | 0.3* | 0.0* | 14.5* | 33.2* | 0.0* | 29.9* |
| | VMRD-Grasp | **14.5** | 0.7* | 0.0* | 6.2* | 10.9* | 4.8* | 8.0* |
| | RynnBrain-Grounding | **83.9** | 0.0 | 10.8 | 11.2 | 59.2 | 0.0 | 76.4 |
| | RynnBrain-Area | 59.4 | 45.3 | 53.2 | 35.8 | **61.5** | 10.1 | 30.9 |

> **深入分析**：
> - 尽管专有模型（如 Gemini 3 Pro、GPT-5.2）拥有数千亿稠密参数与万亿标记的通用知识，但在**物理接地定位任务**（如 RefSpatial、Grasp、RynnBrain-Grounding）中暴露出严重缺陷。例如在 RynnBrain-Grounding 上，GPT-5.2 仅得 11.2，Claude Sonnet 4.5 接近 0.0，而 RynnBrain-30B-A3B 斩获 **83.9** 的绝对断层领先。
> - RynnBrain-30B-A3B 仅以 3B 激活参数的推理开销，便在 VSI-Bench、Open-X VQA、EgoTaskQA 以及 Cornell-Grasp 上全面刷新纪录，实现了算力效率与具身精度的完美统一。

### 6.4 空间思维链推理定量评测：超越前沿思考模型 (Table 5)
为验证物理接地思维链 (CoP) 的有效性，在接触点预测、区域预测和轨迹预测三大核心具身任务上，与开启思考模式 (Thinking Mode) 的前沿大模型进行了系统对决：

| 评估模型 (Model) | 可交互点预测 (Affordance) | 目标区域预测 (Area) | 操作轨迹预测 (Trajectory) | 综合平均分 (Average) |
| :--- | :---: | :---: | :---: | :---: |
| **InternVL3.5-8B** | 63.1 | 9.2 | 47.8 | 40.0 |
| **Qwen3-VL-8B-Thinking** | 56.7 | 20.4 | 46.9 | 41.3 |
| **RoboBrain 2.0-7B** | 65.3 | 38.0 | 58.5 | 53.9 |
| **RoboBrain 2.0-32B** | 73.2 | 39.5 | 60.5 | 57.7 |
| **Qwen3-VL-30B-A3B-Thinking** | 62.2 | 33.0 | 54.8 | 50.0 |
| **GPT-5.2 (Thinking)** | 83.3 | 35.8 | 70.5 | 63.2 |
| **Gemini 3 Pro (Thinking)** | 83.9 | 50.7 | 60.6 | 65.1 |
| **MiMo-Embodied-7B** | 85.3 | 47.1 | 64.9 | 65.8 |
| **RynnBrain-CoP-8B (Ours)** | **90.3** | **59.6** | **71.2** | **73.8** |

> **核心结论**：
> - **突破 90 分大关**：RynnBrain-CoP-8B 在 Affordance 预测上达到 **90.3**，是全行业唯一突破 90 分阈值的模型；
> - **以小博大**：参数仅为 8B 的 RynnBrain-CoP 斩获综合平均分 **73.8**，比 GPT-5.2 (+10.6) 和 Gemini 3 Pro (+8.7) 高出 10 分以上，更以巨大优势超越 32B 的 RoboBrain 2.0 (57.7)。这雄辩证明：**文字交织坐标并引入几何规则强化学习，在物理空间任务中远优于单纯堆叠参数量**。

### 6.5 真实环境与虚拟基准连续导航极限评测 (Table 6)
在连续物理环境视觉语言导航基准 (VLN-CE) 的未知验证集 (Val-Unseen) 上，RynnBrain-Nav-8B 与各大学派经典及最新 SOTA 算法展开了全面评测：

| 导航方法 (Method) | 全景图 (Pano) | 里程计 (Odo) | 深度图 (Depth) | 单目RGB (S.RGB) | R2R 误差 (NE $\downarrow$) | R2R 预言成功率 (OS $\uparrow$) | R2R 成功率 (SR $\uparrow$) | R2R 路径加权 (SPL $\uparrow$) | RxR 误差 (NE $\downarrow$) | RxR 成功率 (SR $\uparrow$) | RxR 路径加权 (SPL $\uparrow$) | RxR 动态规整 (nDTW $\uparrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **VLN-BERT** | $\checkmark$ | $\checkmark$ | $\checkmark$ | | 5.74 | 53.0 | 44.0 | 39.0 | 8.98 | 27.0 | 22.6 | 46.7 |
| **ETPNav** | $\checkmark$ | $\checkmark$ | $\checkmark$ | | 4.71 | 65.0 | 57.0 | 49.0 | 5.64 | 54.7 | 44.8 | **61.9** |
| **ScaleVLN** | $\checkmark$ | $\checkmark$ | $\checkmark$ | | 4.80 | – | 55.0 | 51.0 | – | – | – | – |
| **LAW** | | $\checkmark$ | $\checkmark$ | $\checkmark$ | 6.83 | 44.0 | 35.0 | 31.0 | 10.90 | 8.0 | 8.0 | 38.0 |
| **Seq2Seq** | | | $\checkmark$ | $\checkmark$ | 7.77 | 37.0 | 25.0 | 22.0 | 12.10 | 13.9 | 11.9 | 30.8 |
| **NaVid** | | | | $\checkmark$ | 5.47 | 49.1 | 37.4 | 35.9 | – | – | – | – |
| **NaVILA** | | | | $\checkmark$ | 5.22 | 62.5 | 54.0 | 49.0 | 6.77 | 49.3 | 44.0 | 58.5 |
| **UniNaVid** | | | | $\checkmark$ | 5.58 | 53.3 | 47.0 | 42.7 | 6.24 | 48.7 | 40.9 | – |
| **StreamVLN** | | | | $\checkmark$ | 4.98 | 64.2 | 56.9 | **51.9** | 6.22 | 52.9 | 46.0 | **61.9** |
| **RynnBrain-Nav-8B (Ours)**| | | | $\checkmark$ | **4.92** | **71.6** | **58.6** | 49.6 | **6.20** | **56.1** | **49.6** | 59.6 |

![导航基座能力对比图](https://arxiv.org/html/2602.14979v1/nav.png)
*图 6：对比以 Qwen3-VL 与以 RynnBrain 为基座微调导航模型的性能差距。在不同参数规模下，经过具身时空预训练的 RynnBrain 基座均展现出显著更高的导航成功率与探索上限。*

> **实战亮点**：
> - **仅凭单目 RGB 逆袭多模态输入**：以往顶尖方法（如 ETPNav）高度依赖全景拼接相机、硬件里程计和激光/RGB-D 深度传感器。而 RynnBrain-Nav-8B **仅凭普通单目 RGB 摄像头视频流**，就在 R2R-CE 上斩获 58.6% 的 SOTA 成功率，Oracle 成功率更是飙升至 **71.6%**！
> - **超长程指令遵循韧性**：在指令长、环境复杂的 RxR-CE 上，成功率同样取得 56.1% 的最高分，展现出极强的场景探索拓扑构建与回溯能力。

### 6.6 真实世界多物体操作与多轮规划消融 (Table 7 & Table 8)

#### 1. 多轮规划交互消融实验 (Table 7)
在真实长程家务任务（餐具分发、桌面整理、厨具分发、餐后收餐收拾）中，对比单轮 (Single-Turn, ST) 与多轮动态纠偏 (Multi-Turn, MT) 规划下的任务进程得分 (Task Process, TP $\uparrow$)：

| 规划模型配置 | 分类:易 | 分类:中 | 分类:难 | 整理:易 | 整理:中 | 整理:难 | 餐具:易 | 餐具:中 | 餐具:难 | 收餐:易 | 收餐:中 | 收餐:难 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RynnBrain-Plan-ST 8B** | 72 | 20 | 0 | 60 | 0 | 0 | 34 | 0 | 0 | 90 | 0 | 0 |
| **RynnBrain-Plan-ST 30B** | 75 | 30 | 0 | 58 | 10 | 0 | 28 | 0 | 0 | 95 | 0 | 0 |
| **RynnBrain-Plan-MT 8B** | **100** | **100** | **75** | **100** | 41 | 55 | 92 | 61 | **78** | **100** | 71 | 30 |
| **RynnBrain-Plan-MT 30B** | 85 | 91 | 62 | 84 | **92** | **75** | **95** | **62** | 75 | **100** | **90** | **100** |

消融数据清晰表明：**在复杂多物体环境中，单轮一次性规划在困难 (Hard) 设定下的成功率几乎彻底归零 (0%)**；而引入多轮交互反馈后，30B 模型在严苛的真实 OOD 收餐任务中困难评级成功率达 **100%**。

#### 2. 底层 VLA 真实多物体物理拾取控制评测 (Table 8)

![VLA 评测任务图](https://arxiv.org/html/2602.14979v1/figures/vla_tasks.jpg)
*图 7：RynnBrain-VLA 真实机器人多物体灵巧操作评测场景示意（包含抓取面包、抓取瓷碗以及 6 物体混合复杂场景）。*

评测指标涵盖：抓取动作成功率 (Pickup Success Rate, PSR)、目标识别正确率 (Recognition Success Rate, RSR) 以及端到端总任务成功率 (Success Rate, SR)：

| 控制模型配置 (Method) | 面包:PSR | 面包:RSR | 面包:SR | 瓷碗:PSR | 瓷碗:RSR | 瓷碗:SR | 混合:PSR | 混合:RSR | 混合:SR | 综合:PSR | 综合:RSR | 综合:SR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\pi_{0.5}$-Finetuned** | 0.70 | 0.60 | 0.50 | 0.80 | 0.50 | 0.50 | 0.50 | 0.60 | 0.40 | 0.67 | 0.57 | 0.47 |
| **Qwen3-VL-Finetuned** | 0.70 | **1.00** | 0.70 | 0.50 | **1.00** | 0.50 | 0.60 | **1.00** | 0.60 | 0.60 | **1.00** | 0.60 |
| **RynnBrain-VLA (Ours)** | **0.80** | **1.00** | **0.80** | **0.70** | **1.00** | **0.70** | **0.90** | 0.90 | **0.80** | **0.80** | 0.97 | **0.77** |

相比基于通用基座微调的 $\pi_{0.5}$ (综合 SR 47%) 与 Qwen3-VL (综合 SR 60%)，RynnBrain-VLA 凭借预训练注入的细粒度空间几何与接触点先验，综合成功率达 **77% (0.77)**，抓取稳定性大幅攀升。

---

## 7. 局限性与未来探索方向

尽管 RynnBrain 在具身空间智能的基础模型化上取得了里程碑式的突破，但在迈向通用人机共存与自主智能体的发展中，仍存在若干值得探索的边界：
1. **纯单目视觉下的深度与尺度不确定性**：虽然通过 MASt3R-SLAM 点云预训练显著提升了空间几何感，但在镜面反射、高反光金属器皿、无纹理白墙或强动态晃动视场中，基于归一化像素坐标换算绝对米制深度的精度仍有波动。
2. **终端高频控制的端侧算力功耗比**：尽管 2B 模型和 30B-A3B (激活 3B) 的推理速度表现优异，但在要求 50Hz 以上高频触觉闭环与毫秒级姿态伺服的敏捷灵巧手中，仍需进一步探索量化剪枝与端侧专用 NPU 深度优化。
3. **触觉与力觉 (Tactile & Force Feedback) 的多模态整合**：当前 RynnBrain 主要聚焦于视觉、时空坐标与语言指令。未来的具身基座需进一步吸收高维电子皮肤触觉阵列、关节扭矩传感器数据，实现视-触-力深度融合的世界物理模型。

---

## 8. 核心要点总结 (Key Takeaways)

1. **统一的物理具身基座架构**：RynnBrain 基于 Qwen3-VL 构建了开源具身基础模型家族 (2B, 8B, 30B-A3B MoE)，将离散文本、包围框、交互接触点、功能区域、运动轨迹与抓取姿态统一纳入多模态自回归解码空间，终结了传统 VLM 与动作模型各自为战的割裂局面。
2. **创新的物理接地思维链 (CoP)**：通过将文字思考步骤与视频几何定位图元交织，并辅以纯规则几何强化学习 (PPO/GRPO) 奖励驱动，彻底消除了纯文本 CoT 在空间推理上的漂移与幻觉，以 8B 规模在接触点与轨迹推断上力压 GPT-5.2 与 Gemini 3 Pro。
3. **工业级 2000 万时空预训练飞轮**：融合 SLAM 3D 点云重构、重力对齐世界坐标系与长时序第一人称视频交互数据，构建了跨场景泛化的高保真物理世界常识，且完全不损害通用图文视频理解能力。
4. **全链路可部署的生态闭环**：派生的 RynnBrain-Nav 刷新单目视觉连续导航 SOTA；RynnBrain-Plan 攻克长程复杂家务的分层多轮动态调度；RynnBrain-VLA 大幅提升多物体物理抓取成功率。全栈采用 Apache 2.0 协议开源，为具身智能实体化落地提供了强大的公共技术底座。
