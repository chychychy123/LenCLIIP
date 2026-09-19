# LenCLIP

Foreground-Background Text Contrastive Learning for Weakly Supervised Semantic Segmentation

根据提供的论文和基线源码重建的 LenCLIP 研究实现，支持 PASCAL VOC 2012 与 MS COCO 2014。项目入口、模型包、配置、日志和检查点统一使用 LenCLIP 命名。

**交付状态：按要求未执行训练、推理、单元测试、导入测试或语法编译测试。此版本是根据论文重建的源码，不是找回的原始实验代码，也不声称已复现论文指标。** 论文未明确的细节及公式处理见 [实现说明](docs/IMPLEMENTATION.md)。压缩包不含图像数据、CLIP 预训练权重或训练后的模型。

## 1. 方法与源码

| 部分 | 实现 | 功能 |
| --- | --- | --- |
| MG | `LenCLIP_model/lpg.py`、`lenclip.py` | 学习类别掩码；将原始 RGB 分成前景/背景，经过冻结 CLIP 得到区域特征；计算匹配损失 |
| LP | `LenCLIP_model/prompts.py`、`lpg.py` | 三套可学习上下文：MG 类别提示、前景提示、背景提示；长度均为 30 |
| VA | `LenCLIP_model/visual_alignment.py` | 类别语义引导的局部特征传播及 CAM 生成 |
| 分割 | `segformer_head.py`、`Decoder/TransDecoder.py` | 多层 CLIP 特征融合与 Transformer 分割解码 |
| 在线伪标签 | `LenCLIP_model/lenclip.py`、`PAR.py` | 类别过滤、注意力传播、背景分数、PAR 细化 |
| 数据 | `datasets/lenclip_data.py` | 训练仅读取 RGB 与图像级标签；像素标注只用于评估 |

CLIP 的图像/文本编码器权重均冻结，但保留从编码器输出到可学习提示和图像掩码的梯度。优化器包含提示参数、掩码生成器、特征融合适配器与分割解码器。

## 2. 环境安装

以下是源码对应的安装配方，尚未在本次交付中运行。可使用 Linux 或 Windows；训练建议使用 NVIDIA CUDA GPU。

```bash
conda create -n LenCLIP python=3.10 -y
conda activate LenCLIP
cd /path/to/LenCLIP
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

CUDA wheel 应与本机驱动兼容；上面给出 CUDA 12.1 的示例。项目已包含修改后的 `clip` 包，不需要另装同名 CLIP 包，也不依赖 MMCV、MMSeg 或 Grad-CAM。所有运行命令均在解压后的 `LenCLIP/` 根目录执行。

## 3. CLIP 预训练权重

从 OpenAI CLIP 官方地址下载 ViT-B/16，保存到 `pretrained/ViT-B-16.pt`：

```bash
curl -L https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt -o pretrained/ViT-B-16.pt
```

Windows PowerShell 中将上面的 `curl` 写成 `curl.exe`。也可浏览器下载后放入该目录。训练及推理初始化都会读取此文件。

## 4. 数据准备

压缩包已带有原工程的划分列表和 `cls_labels_onehot.npy`。VOC 图像级标签长度为 20，COCO 为 80，均不含背景。标签顺序见 `clip/clip_text.py`；分割 ID 为背景 0、前景 1..20 或 1..80，忽略像素 255。

### PASCAL VOC 2012

```text
LenCLIP/
  data/VOCdevkit/VOC2012/
    JPEGImages/                 # 训练及推理图像
    SegmentationClass/          # val 评估用的原始类别索引 PNG
    SegmentationClassAug/       # 可选：仅在评估时用作缺失标注的回退目录
  datasets/voc/
    train_aug.txt
    train.txt
    val.txt
    test.txt
    cls_labels_onehot.npy
```

训练默认使用 `train_aug`。请确保 `JPEGImages` 覆盖列表中所有图像，包括增强训练集的图像；仅有 VOC trainval 压缩包时可能仍需补充 SBD 图像。训练加载器不读取 `SegmentationClassAug`，无需以像素标注生成训练标签。默认训练会周期评估 `val`，因此需要验证集标注；不需要评估时设置 `train.eval_iters=0`。

### MS COCO 2014

支持以下两种图像摆放方式之一：

```text
data/MSCOCO/
  JPEGImages/train/             # COCO_train2014_000000000009.jpg 等
  JPEGImages/val/               # COCO_val2014_000000000042.jpg 等
  SegmentationClass/val/        # 验证集索引 PNG
```

或直接使用 `data/MSCOCO/train2014/` 与 `data/MSCOCO/val2014/`。保留官方完整图片文件名，与 `datasets/coco/*.txt` 一致。训练不需要 COCO 分割 PNG。

评估掩码需为连续类别 ID 1..80，不能直接使用稀疏的 COCO category_id。若已有和实验协议一致的索引掩码，请直接使用。也提供可选转换脚本：

```bash
python -m pip install -r requirements-coco.txt
python scripts/prepare_coco_masks.py --annotations data/MSCOCO/annotations/instances_val2014.json --output data/MSCOCO/SegmentationClass/val
```

转换规则是小实例覆盖大实例、未被实例覆盖的 crowd 像素设为 255，详见实现说明。此转换规则未被论文指定，比较指标前应统一评估掩码协议。脚本默认跳过已有文件，只有指定 `--overwrite` 才覆盖。

## 5. 配置路径与参数

修改 `configs/lenclip_voc.yaml` 或 `configs/lenclip_coco.yaml`：

```yaml
dataset:
  root_dir: D:/Datasets/VOCdevkit/VOC2012
  name_list_dir: datasets/voc
model:
  clip_pretrain_path: pretrained/ViT-B-16.pt
train:
  work_dir: LenCLIP_runs/voc
```

路径支持绝对路径；配置中的相对数据、列表、权重和输出路径统一相对项目根目录解析。Windows YAML 路径建议用 `/`。上面是字段示例，请修改完整配置文件中的对应字段。

论文明确的默认参数为 `prompt.n_ctx=30`、`loss.lambda_a=0.6`、`loss.lambda_b=0.2`、`loss.lambda_lp=0.7`、AdamW 学习率 `1e-4`、权重衰减 `1e-2`，推理尺度 `[0.75, 1.0]`。训练迭代数沿用基线设置：VOC 30000、COCO 80000。

## 6. 训练与恢复

```bash
# VOC
python scripts/train_lenclip.py --config configs/lenclip_voc.yaml --device cuda

# COCO
python scripts/train_lenclip.py --config configs/lenclip_coco.yaml --device cuda

# 指定数据目录、减小 batch，并关闭训练中的验证
python scripts/train_lenclip.py --config configs/lenclip_voc.yaml --set dataset.root_dir=D:/Datasets/VOCdevkit/VOC2012 train.batch_size=1 train.num_workers=0 train.eval_iters=0

# 指定输出目录
python scripts/train_lenclip.py --config configs/lenclip_voc.yaml --work-dir LenCLIP_runs/voc_experiment

# 从完整检查点继续；使用该实验保存的配置
python scripts/train_lenclip.py --config LenCLIP_runs/voc/LenCLIP_config.yaml --resume LenCLIP_runs/voc/checkpoints/LenCLIP_iter_2000.pth
```

含空格的命令行路径需要引号，例如 `--set "dataset.root_dir=D:/My Datasets/VOC2012"`。Windows 如遇 DataLoader 多进程问题，可设置 `train.num_workers=0` 和 `inference.num_workers=0`。

训练是单进程、单设备入口。CUDA 设备可通过 `--device cuda:1` 选择。掩码分支需要额外的 CLIP 前向/反向，因此显存开销高于单纯分割解码。默认开启梯度检查点；显存不足可先减小 `train.batch_size` 和 `model.region_batch_size`。`pseudo.par_scale=0.5` 可降低细化内存，但会改变伪标签细化分辨率。

输出示例：

```text
LenCLIP_runs/voc/
  LenCLIP_config.yaml
  LenCLIP_train.log
  LenCLIP_tensorboard/
  LenCLIP_metrics_2000.json       # 开启验证时产生
  checkpoints/
    LenCLIP_iter_2000.pth
    LenCLIP_best.pth             # 开启验证且得到有效指标时产生
```

每个定期检查点包含完整模型、优化器、迭代数、配置和随机状态。恢复会从保存迭代继续，但数据加载器从新迭代器开始，不保证逐批完全一致。最后一轮总会保存检查点。

```bash
tensorboard --logdir LenCLIP_runs/voc/LenCLIP_tensorboard
```

## 7. 多尺度推理与评估

推理只输入图像，不需要图像级标签。默认尺度为 `[0.75, 1.0]`，翻转默认关闭；`--flip` 可开启额外水平翻转。

```bash
# VOC val：保存预测并计算 mIoU
python scripts/infer_lenclip.py --config LenCLIP_runs/voc/LenCLIP_config.yaml --checkpoint LenCLIP_runs/voc/checkpoints/LenCLIP_iter_30000.pth --split val --evaluate --output LenCLIP_predictions/voc

# COCO val
python scripts/infer_lenclip.py --config LenCLIP_runs/coco/LenCLIP_config.yaml --checkpoint LenCLIP_runs/coco/checkpoints/LenCLIP_iter_80000.pth --split val --evaluate --output LenCLIP_predictions/coco

# 单张图片，不需要任何标签
python scripts/infer_lenclip.py --config configs/lenclip_voc.yaml --checkpoint LenCLIP_runs/voc/checkpoints/LenCLIP_iter_30000.pth --input examples/image.jpg --output LenCLIP_predictions/example

# 整个图片目录；可额外启用翻转
python scripts/infer_lenclip.py --config configs/lenclip_voc.yaml --checkpoint LenCLIP_runs/voc/checkpoints/LenCLIP_iter_30000.pth --input D:/Pictures --scales 0.75 1.0 --flip --output LenCLIP_predictions/images

# VOC test 只输出预测，不带 --evaluate；需要另行准备 test 对应图像
python scripts/infer_lenclip.py --config configs/lenclip_voc.yaml --checkpoint LenCLIP_runs/voc/checkpoints/LenCLIP_iter_30000.pth --split test --output LenCLIP_predictions/voc_test
```

`examples/image.jpg` 是待替换的示例路径，包内未包含该图片。预测为带调色板的 PNG，文件中的像素值仍是类别 ID。数据集模式保存为 `<image_id>.png`；任意图片模式保存为 `<原文件名含扩展名>.png`，避免同名 JPG/PNG 相互覆盖。评估结果写入 `LenCLIP_metrics.json`，mIoU 与各类 IoU 的取值范围为 0..1，显示百分数时乘以 100。

## 8. 实现边界与来源

论文中背景提示监督、余弦对数定义域、VA 权重和语义细化方式没有完整定义，因此本实现提供了明确的默认补全，并保留相应配置开关。请阅读 [论文到代码对应与差异](docs/IMPLEMENTATION.md)，不要把补全参数误当成原始实验设置。

保留和改写的第三方代码来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。目录与运行品牌统一为 LenCLIP，不改变原有代码的来源归属。
