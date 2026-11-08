# I2I PR 条件流匹配替换

这是对 I2I-PR 的直接速度监督替换代码整理副本，包含匹配 InDI、FM、多步 Euler 与端点物理一致性。代码从原实验中分离，未上传数据、权重、机器连接信息或原始运行日志。当前仅完成发布前准备，没有创建仓库或推送。

## 方法

10个HIO候选均值为m，干净训练图为y，源点为z0=m+(3/255)epsilon。tau=t/32，x=tau*z0+(1-tau)*y，前进时间s=1-tau。11通道输入是x和10个候选。

- InDI：D=q+mean([x,h])，主要监督端点MSE。
- FM：v=q，D=x+tau*v，直接监督v接近y-z0，没有tau平方权重。
- 两臂都加入0.5权重的DC后端点监督，全部损失乘255平方；这不是纯速度风险。
- 每步先修正干净端点再推进。FM速度修正为v+(P(D)-D)/tau。
- 两臂共享骨干和采样预算，但零头初始端点不同；不能把差异只归因于单一损失变化。

主实验4NFE、每阶段5DC；另保留1/8NFE与固定总40DC。这里不是FLOWER重加噪实现，也不提供相位恢复后验保证。

## 文件与入口

| 文件 | 用途 |
|---|---|
| full_fm_core.py | 直接速度/端点参数化、损失、DC与积分；仅替换了原环境导入 |
| curriculum_train_pair.py | 原训练入口，保留断点状态、公共课程与验证选模；原文件字节不变 |
| independent_test_evaluate.py | 原固定256来源、57配置评价入口；仅替换环境导入 |
| phase_metrics.py | 原始、平移/翻转歧义和测量残差指标；原文件字节不变 |
| upstream_adapter.py | 从另行取得的官方源码加载固定类和物理算子，无下载行为 |
| preflight.py | 不导入torch、不访问GPU的只读环境与文件检查 |
| configs/train.example.json | 无个人路径的训练配置模板，数值审计标记故意为false |
| upstream_sources.json | 外部源码版本与文本哈希 |
| docs/PROVENANCE.json | 原实验及整理后文件哈希和改动说明 |
| docs/DATA_CONTRACT.md | 私有配对缓存、manifest和评价协议接口 |
| docs/RESULTS.md | 已有独立测试的匿名汇总及限制 |
| NOTICE.md | 论文、上游来源与尚未确认的授权范围 |

## 外部依赖与准备

上游快照与官方仓库根目录未发现LICENSE。这里不重新分发上游源码、notebook或权重，也不擅自给整体附MIT等许可。需要由使用者另行确认上游使用、修改和衍生发布的授权范围。仅配置为私有仓库也不能自动解决许可问题。

需要单独取得官方仓库 https://github.com/METU-SPACE-Lab/I2I-PR-for-Phase-Retrieval 的记录提交091950820cec58c70853d89862143dabcf65661d，将I2IPR_UPSTREAM设为含utils与models的源码根目录。适配器逐文件检查文本哈希，只提取UNet2DModel11Channels类，不执行notebook及其他可能加载私人权重的模型类。

建议Python3.10及匹配CUDA的PyTorch；requirements是依赖约束，不是完整运行环境锁。diffusers等三个固定版本来自既有离线依赖记录。当前Windows没有torch和diffusers，未安装或修改环境。原训练使用CUDA、BF16、float64 DC，约2.55亿参数、批次4；本项目获授权计算设备仅限既有RTX3090环境。

先运行 `python preflight.py --upstream <官方源码目录>`。它只检查语法、包元数据与源码哈希，不证明GPU数值或训练可运行。该检查不启动计算。

训练模板不能直接运行：需要合法数据的配对缓存、许可和分组去重审计、manifest哈希、固定验证身份哈希，以及对应代码和目标环境的DC反传数值检查。不要把旧版代码审计直接冒认为新适配器已验证，也不要只把false改成true来绕过检查。

完成上述条件且另有计算授权后，训练入口为 `python curriculum_train_pair.py --config <本地完整配置> --manifest <本地清单> --mode fm --seed 2027 --out <新输出目录> --stop-after 5000`。匹配臂将mode改为indi。第二阶段还需要公共transition记录、边界检查点和resume路径。所有数据路径与检查点均应放在仓库之外。

`independent_test_evaluate.py --help`列出原评价协议参数。它要求锁定检查点、协议、训练及测试manifest和审计文件，固定256来源，不是任意单图演示入口。此副本没有提供真实测试清单、权重或自动采集数据。只有显式授权新实验时才运行，不能对已揭盲数据自动重测或挑选配置。

## 已有结论

每方法两种子、每臂30,000更新；共同课程914来源扩至5,000来源，固定49验证，独立测试256来源。预定4NFE20DC中FM减InDI为+0.094300和-0.183717dB，未稳定胜出。详见docs/RESULTS.md。整理版没有重新训练或重跑实验；导入适配改变后的GPU数值与吞吐仍需后续验证。

## 发布状态

这是独立审查副本，没有Git历史、remote或自动推送脚本。尚需用户选定目标仓库与公开范围，并确认项目代码许可和上游衍生使用边界。任何凭据、SSH配置、私有源身份、图片、缓存、日志与权重都不属于发布清单。
