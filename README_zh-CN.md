# VC-SAM2-UNet

**面向多标签胎儿心脏超声分割的视图条件 SAM2-UNet**  
论文稿件面向 IECBES 2026。本仓库不声称论文已录用、发表或拥有 DOI。

**作者：** Ziqi Zhao、Zhenghao Jiang、Camilla Fazi、Netzahualcoyotl Hernandez-Cruz

**机构：** 西交利物浦大学人工智能与先进技术学院；佛罗伦萨大学神经科学、心理学、药物研究与儿童健康系；佛罗伦萨 Careggi 大学医院新生儿科。Ziqi Zhao 与 Zhenghao Jiang 贡献相同；Netzahualcoyotl Hernandez-Cruz 为通讯作者。

> 当前仓库为公开的研究代码仓库。数据、权重和原始 SAM2 运行环境未随仓库提供，完整复现状态见[复现说明](docs/reproduction_status.md)。

## 方法概览

VC-SAM2-UNet 在 SAM2-UNet 上加入两个模块：残差 Metadata-FiLM（RMF）和 Global Selective Dual-Path BiFPN（GSDB）。模型面向 4CHV、LVOT、RVOT、3VTV 四类标准视图，输出 14 个可重叠的独立分割通道。

- A0：SAM2-UNet baseline；
- A1：仅 Metadata-FiLM，分割训练使用 GT view；
- A2：仅 Selective DPBiFPN，不输入 view；
- A3：完整 VC-SAM2-UNet，GT-view 训练；
- Auto-Hard/Auto-Soft：同一个 A3 分割 checkpoint，分别使用冻结 ResNet-18 的 argmax 类别或四类概率。

![VC-SAM2-UNet 架构图](assets/vc_sam2_unet_architecture.png)

该图按用户提供的论文插图原样收录，含论文中的超声输入示例和合成 mask 可视化；仓库未包含数据集、病例编号或模型权重。

## 结果

Validation 为每种配置 10 个 seed 的均值 ± 样本标准差。固定 test 使用各自根据 validation 选出的 SeedMax checkpoint，共 179 张图。

| 配置 | Val Dice（10 seeds） | SeedMax | 最佳 epoch | Test Dice | Test IoU |
|---|---:|---:|---:|---:|---:|
| SAM2-UNet baseline | 0.7878 ± 0.0048 | 9102 | 70 | 0.7908 | 0.6616 |
| + Metadata-FiLM（RMF） | 0.8231 ± 0.0024 | 5772 | 39 | 0.8135 | 0.6910 |
| + Selective DPBiFPN（GSDB） | 0.7874 ± 0.0046 | 1337 | 68 | 0.7998 | 0.6734 |
| + 两组件（Full） | 0.8204 ± 0.0051 | 1337 | 71 | 0.8214 | 0.7021 |

Full 相对 baseline 的固定 test Dice 差值是 `0.8214 - 0.7908 = 0.0306`，即 **3.06 个 Dice 百分点**，不是相对提升 3.06%。

| A3 视图模式 | 分割 seed | Test 图像数 | Dice | IoU |
|---|---:|---:|---:|---:|
| GT View | 1337 | 179 | 0.8214 | 0.7021 |
| Auto-Hard | 1337 | 179 | 0.8131 | 0.6900 |
| Auto-Soft | 1337 | 179 | 0.8065 | 0.6815 |

机器可读汇总位于 `results/`。这些 IoU 来自实际 evaluation 汇总，不是由 Dice 公式换算得到。

可绘制汇总 test 结果图（不读取图像、mask 或 checkpoint）：

```bash
python visualization/plot_paper_results.py --output /path/to/figures/fixed_test_summary.png
```

## 训练和评估协议

- 893 张图，划分为 train/validation/test = 625/89/179；
- 14 个独立 sigmoid 通道，允许掩膜重叠；
- 灰度转 RGB，分割图直接双线性缩放到 352×352，掩膜使用最近邻；
- ImageNet 均值/标准差归一化；
- 逐图逐通道 validity mask 排除部分标注中的未知类别；
- 100 epoch、batch size 2、AdamW、余弦学习率；final 与两个 side output 使用 masked BCE+Dice 深监督；
- validation macro Dice 选择 best checkpoint；固定 test 阈值 0.5；不使用 TTA、校准、VAP、阈值调参或后处理；
- evaluation 使用 `model.eval()`、`shuffle=False`、`drop_last=False`。

分类器使用 224×224 保纵横比缩放和补零，与分割预处理不同。具体参数见 `configs/paper_protocol.json`。

## 安装与运行

原 XEC 环境为 Python 3.10.20、PyTorch 2.5.1+cu124（CUDA build 12.4）、torchvision 0.20.1+cu124。训练环境调用的是项目内第三方 SAM2 fork（commit `c2549321706fa290b10f9e94227b71ebf7512524`），因其许可无法核实，故不随仓库发布。下方安装命令使用固定版本的官方 Meta SAM 2 作为有 Apache-2.0 许可的替代项，**尚未验证与原 fork 等价或兼容原 checkpoint**。详见 `docs/environment.md`。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
```

数据集、train/val/test 清单、SAM2 权重、分割 checkpoint 和冻结 View classifier checkpoint 均不随仓库提供。授权后参考英文 README 中的命令，分别运行 Baseline/Full、A1/A2、View classifier 和 fixed-test evaluation。

## 隐私、许可和免责声明

仓库不含超声图像、mask、病例清单、患者元数据、checkpoint、预测结果、SSH 凭据或 HPC 路径。数据合同见 `docs/data_contract.md`。目前没有项目级 LICENSE；未经全体作者及机构确认，不添加 MIT 或其他开源许可。第三方许可文本保留在 `third_party/licenses/`。

本代码仅用于研究和教育，不是医疗器械，不能用于诊断、治疗或临床决策。

仓库收录了用户提供的论文架构图（其中含一个超声输入示例和合成 mask 可视化）；未单独打包数据集图像、病例级定性结果、Grad-CAM 或 confusion figure。
