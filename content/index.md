---
title: 具身智能知识花园 (Quartz 预览)
---

# 🌱 具身智能知识花园 (Quartz 预览)

欢迎来到基于 **Quartz** 搭建的 Obsidian 知识花园！

## 📄 论文精读

- [[paper_2512.01031/VLASH_论文解读|VLASH: Real-Time VLAs via Future-State-Aware Asynchronous Inference]]
  - **标签**：#具身智能 #VLA #实时控制 #异步推理 #未来状态感知 #动作分块 #论文解读
  - **亮点**：MIT 韩松团队首创未来状态感知 (Future-State-Aware) 异步推理范式，通过已知动作前滚状态彻底消除预测-执行时序错位；配合共享观测微调提速 3.26 倍，将真机反应延迟骤降最高 17.4 倍，首破 VLA 连续打乒乓球与打地鼠等极限动态交互。
- [[paper_2603.19199/FASTER_论文解读|FASTER: Rethinking Real-Time Flow VLAs]]
  - **标签**：#具身智能 #VLA #实时控制 #流匹配 #动作分块 #论文解读
  - **亮点**：港大赵恒爽团队颠覆流匹配 VLA 恒定时间步采样范式，提出视界感知调度 (HAS) 与流式即解即发架构；首动作时间 (TTFA) 提速超 3 倍，消费级显卡 (RTX 4060) 亦可赋能 30Hz 高动态乒乓球接球反击与日常灵巧操作，刷新动态基准 SOTA。
- [[paper_2602.16710/EgoScale_论文解读|EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data]]
  - **标签**：#具身智能 #VLA #灵巧操作 #人类视频学习 #缩放定律 #跨构型迁移 #论文解读
  - **亮点**：英伟达与伯克利团队推进 20,854 小时第一人称视频预训练，首次确立灵巧操作对数线性 Scaling Law ($R^2=0.9983$)；提出“规模预训练+轻量中训”两阶段解耦范式，真机成功率飙升 54%，仅凭 1 次示范实现折叠衣物等单样本涌现泛化，并成功跨构型赋能 G1 人形机器人三指手。
- [[paper_2511.17502/RynnVLA-002_论文解读|RynnVLA-002: A Unified Vision-Language-Action and World Model]]
  - **标签**：#具身智能 #VLA #世界模型 #动作世界模型 #论文解读 #机械臂操作
  - **亮点**：阿里达摩院首创动作世界模型 (Action World Model) 统一架构，攻克自回归动作误差级联难题；无预训练下 LIBERO 达到 97.4% SOTA，真机 LeRobot SO100 抗干扰成功率超传统基线 30%。
- [[paper_2608.26067/StreamPI_论文解读|StreamPI: Streaming Multimodal Temporal Modeling for Vision-Language-Action Models]]
  - **标签**：#具身智能 #VLA #时序建模 #流式推理 #机器人操作 #论文解读
  - **亮点**：首创指令锚定式时序建模与原子时序单元，零额外参数全面激活 VLA 时序与空间几何感知；结合随机间隔训练与增量 KV Cache，真机 80% 破解三仙归洞猜球与动态抓取，LIBERO 达 98.3% 刷新纪录。
- [[paper_2511.00091/PLD_论文解读|Self-Improving Vision-Language-Action Models with Data Generation via Residual RL (PLD)]]
  - **标签**：#具身智能 #VLA #强化学习 #残差控制 #自改进飞轮 #论文解读
  - **亮点**：首创 Probe, Learn, Distill (PLD) 三阶段自改进范式，通过轻量残差专家与主动探测生成部署对齐纠错轨迹，LIBERO 达到 99.2% 饱和成功率，实现 1 小时真实 GPU 拔插装配无人工干预。
- [[paper_2507.12440/EgoVLA_论文解读|EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos]]
  - **标签**：#具身智能 #VLA #论文解读 #人形机器人 #灵巧操作
  - **亮点**：利用 80 亿人类日常视频，构建基于 MANO 动作先验的双臂灵巧操作大模型。

## 💡 特性演示
- **快捷键**：按下 `Ctrl + K` 唤起 Spotlight 实时全局搜索
- **右侧面板**：可交互的网状关系图谱 (Graph View) 与大纲目录
- **标签归类**：支持点击任意标签进入专属分类聚合页
