# 来源归属与许可核查

研究基础：Mehmet Onurcan Kaya、Figen S. Oktem，I2I-PR: Deep Iterative Refinement for Phase Retrieval using Image-to-Image Diffusion Models，arXiv:2507.09609，首次2025年，v2为2026年。

- 论文：https://arxiv.org/abs/2507.09609
- 上游：https://github.com/METU-SPACE-Lab/I2I-PR-for-Phase-Retrieval
- 记录提交：091950820cec58c70853d89862143dabcf65661d

替换实验继承了上游的模型结构、傅里叶尺度与HIO约束设计；当前适配器依赖使用者另行取得的上游文件，不包含其源码。项目新代码包括速度监督、匹配比较、训练与评价编排；出处和对应文件哈希见docs/PROVENANCE.json。不声称独立从零发明整个I2I-PR框架。

2026年11月7日的本地快照及GitHub根目录检查未发现LICENSE文件。未据此认定有再分发或衍生开源授权；也未擅自选择新代码的公开许可证。请在发布前确认授权与拟采用的项目许可。上游许可端点不在当前连接器允许的读取范围内，因此该端点未能进一步验证。

依赖由各发行方提供，应分别保留和遵守其许可证。此整理副本不捆绑任何依赖wheel、权重、第三方图像或上游notebook。论文许可不能替代代码许可，数据来源许可也不能替代模型权重许可。
