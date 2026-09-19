# LenCLIP：论文到实现的对应说明

依据：用户提供的五页论文 *Foreground-Background Text Contrastive Learning for Weakly Supervised Semantic Segmentation*，重点是第 3 节、图 2、第 4.1 节与参数分析。

本工程是在提供的源码基础上重建的方法实现。论文没有提供足够细节来唯一恢复原始代码，因此下文区分明确给出的公式、沿用的基线组件以及本次补全。没有执行任何训练、推理、单元测试、导入测试或编译测试，也没有产生可报告的复现指标。

## 一、数据流与梯度

1. 数据加载器输出 `images: [B,3,H,W]`，值域 0..1；`class_labels: [B,C]` 为图像级多热标签，C 不含背景；裁剪填充区另有 `valid_mask`。
2. 冻结的 CLIP ViT-B/16 输出各层视觉 token。前 11 层经过原有 MLP 融合适配器，得到 `[B,256,H/16,W/16]`；三层 Transformer 解码器产生 `[B,C+1,H/16,W/16]` 分割 logits。
3. MG 使用融合特征预测 C 个类别掩码。针对图像级标签中的正类，构造 `X_f=M*X` 和 `X_b=(1-M)*X`，分别经过 CLIP 图像编码器得到单位化区域向量。
4. MG、前景、背景三套可学习上下文经过冻结文本编码器得到 `v`、`v_f`、`v_b`。上下文必须经过有梯度的编码器计算，不能预计算为固定文本特征。
5. 区域向量和提示计算 LPG 损失；梯度通过文本编码器传回上下文，通过区域图像编码器传回掩码生成器及融合适配器。CLIP 参数保持 `requires_grad=False`。
6. VA 对 CLIP 最后一层经过 `ln_post` 和 `proj` 的 patch 特征进行语义引导局部传播，形成 CAM。伪标签经类别过滤、注意力传播及 PAR 细化后 `argmax`。
7. 分割与亲和损失使用在线伪标签。伪标签分支停止梯度；其类别集合直接来自图像级标签，不从像素标注推导。

训练时读取的文件仅为图像、划分列表和图像级标签字典；验证器使用独立数据集读取分割标注，推理时不读取图像级标签。

## 二、MG：论文公式 (1)

文件：`LenCLIP_model/lpg.py`、`LenCLIP_model/lenclip.py`。

```text
s_f = cosine(u_f, v)
s_b = cosine(u_b, v)
L_match = softplus(s_b - s_f + log(lambda_a))
```

这是 `-log(exp(s_f)/(exp(s_f)+lambda_a*exp(s_b)))` 的数值稳定写法，没有增加温度项。对一个 batch 中所有存在的“图像、类别”对取平均。

论文未给出 `S(X)` 的网络结构。本实现采用冻结 CLIP 特征 + 融合适配器 + `3x3 Conv/GELU/1x1 Conv/Sigmoid`；掩码双线性上采样到输入大小。先在原始 0..1 RGB 上乘掩码，再做 CLIP 归一化，避免把归一化空间中的零误当成黑色背景。

区域编码采用经过全部 12 个视觉块的 CLS token，并通过 `ln_post/proj` 投影到文本空间。默认将区域图像缩放到 224x224；这是本次实现的计算量折中，论文没有给出区域输入分辨率。可用 `model.region_size=320` 更改。`model.region_batch_size` 控制每次编码的正类对数量，每对包含前景与背景两张图；不截断正类集合。

## 三、LP：论文公式 (2)-(4) 及两个未定义问题

文件：`LenCLIP_model/prompts.py`、`LenCLIP_model/lpg.py`。

每条提示为 `[SOS, 30 个可学习 context token, 类名, '.', EOS, padding]`，CLIP 文本总长度仍为 77。占位字符 X 的嵌入全部替换为可学习参数，不作为固定语义模板。三组上下文分别初始化，默认在类别间共享，同一个上下文与不同类名组合；`prompt.class_specific=true` 可启用每类独立上下文。共享方式、初始化标准差 0.02 和标点均为实现补全。

### 3.1 余弦相似度进入 log 的定义域

论文直接写 `L_f=-log(cos(u_f,v_f))` 与 `L_b=-log(1-cos(u_b,v_f))`，但余弦值可以为负。源码提供两种明确模式：

| 配置 | 定义 | 影响 |
| --- | --- | --- |
| `loss.cosine_mode=shifted`（默认） | `p=clamp((cos+1)/2, eps, 1-eps)` | 让 log 输入落在概率区间；负余弦区域仍保留梯度，但与论文的原始数值公式有差异 |
| `loss.cosine_mode=clamped` | `p=clamp(cos, eps, 1-eps)` | 尽量按字面公式计算；余弦落在裁剪区间外时梯度为零 |

公式 (1) 始终使用原始余弦，不受此开关影响。

### 3.2 论文定义了背景提示，却没有在公式中使用它

论文文字定义 `v_b=E_T(t_b)`，但公式 (2)、(3) 只出现 `v_f`，公式 (4) 因此不能直接训练背景提示。图 2 的部分前景/背景标注也与正文不一致。这里以正文的前景定义为准，并将背景分支的补全单独列出：

```text
L_f = -mean(log(p(cos(u_f, v_f))))
L_b = -mean(log(1 - p(cos(u_b, v_f))))
L_LP_paper = L_f + lambda_b * L_b

L_bg_completion = -mean(log(p(cos(u_b, v_b))))
                  -lambda_b * mean(log(1 - p(cos(u_f, v_b))))
L_LP = L_LP_paper + background_prompt_weight * L_bg_completion
```

默认 `background_prompt_weight=1.0`，使背景提示获得与前景对称的监督。这是为了落实论文的“双提示学习”叙述所作的额外选择，**不是论文明确给出的公式，也未经消融验证**。日志分别记录 `lp_paper` 和 `background_completion`。

如果需要按字面损失公式进行实验，可使用：

```bash
python scripts/train_lenclip.py --config configs/lenclip_voc.yaml --set loss.cosine_mode=clamped loss.background_prompt_weight=0 va.background_scale=0
```

该设置关闭背景补全及背景提示对 CAM 的影响，背景上下文不再获得有效监督。它只用于比较字面损失定义，不代表可以恢复论文中的双提示结果。

## 四、VA：公式 (5) 与语义细化

文件：`LenCLIP_model/visual_alignment.py`。

论文只写出了局部加权求和，没有定义 `alpha` 的计算、邻域大小、迭代数或 `T_hat` 的产生方式。本实现使用：

```text
A_k(p) = cosine(F(p), v_f[k])
score_k(p,q) = cosine(F(p), F(q)) / visual_temperature
               - abs(A_k(p)-A_k(q)) / semantic_temperature
alpha_k(p,q) = softmax_q(score_k(p,q)), q in N(p)
F_hat_k(p) = normalize(sum_q alpha_k(p,q) F(q))
T_hat_k = normalize(v_f[k] - background_scale * v_b[k])
CAM_k(p) = ReLU(cosine(F_hat_k(p), T_hat_k))
```

每个类别的 CAM 按空间最大值归一化。默认邻域半径 1（3x3，含中心），一次传播，两项温度均为 0.1，背景语义扣除系数 0.2；边缘邻域以有效位置掩码排除补零位置。使用 `unfold` 提取局部邻域，逐图、逐正类处理，没有构造每类完整的 N×N VA 矩阵。

`va.enabled=false` 关闭特征传播，但保留提示生成的 CAM；`va.background_scale=0` 关闭 `T_hat` 中的背景扣除。上述公式与默认超参数均属于论文未指定部分的补全，并非恢复出的原始知识库实现。

## 五、公式 (6) 与分割框架的联合目标

论文公式 (6) 描述的是 MG/LP 目标；如果仅优化它，分割解码器没有监督。因此保留原有在线分割训练目标：

```text
L_LPG = L_match + lambda_lp * L_LP
L_train = seg_weight * L_seg
          + affinity_weight * L_aff
          + lpg_weight * L_LPG
```

默认 `seg_weight=1`、`affinity_weight=0.1`、`lpg_weight=1`。`L_seg` 对前景、背景分别求交叉熵再平均，忽略裁剪填充区；当其中一组为空时只计算非空组，避免空集合导致 NaN。`L_aff` 对局部正负像素对分别计算误差，排除填充像素和对角自关联。

论文第 4.1 节写“仅训练 adapter 和 decoder”，而第 3 节、图 2 又要求训练掩码与提示。本实现遵从方法章节：CLIP 主干冻结，提示、掩码生成器、融合适配器、解码器均进入优化器。所有可学习组件使用同一学习率 1e-4，不保留基线分割头的额外十倍学习率。

## 六、沿用和调整的基线部分

| 部分 | 处理 |
| --- | --- |
| 多层特征融合 | 保留前 11 层 MLP 投影、拼接、1x1 融合和 dropout |
| 分割解码器 | 保留三层 pre-norm Transformer 与 1x1 类别预测 |
| 注意力算子 | 使用 PyTorch `nn.MultiheadAttention`，保留预训练权重的参数名布局 |
| CLIP 前向 | 删除编码器内部无条件 `no_grad`；始终执行全部块，不再凭 token 数等于 77 来区分视觉/文本 |
| CLIP 特征投影 | 视觉/文本均使用各自原始 LayerNorm 和 projection；相似度在公共嵌入空间计算 |
| 在线 CAM | 从固定模板 Grad-CAM 改为 LP + VA 余弦 CAM |
| 伪标签传播 | 平均 CLIP 后 8 层 patch 注意力，15000 轮后乘解码融合特征的 sigmoid 亲和矩阵，再做行归一化；该策略保留注意力细化思路，但没有完全照搬旧 CAM 的包围框筛选和注意力层筛选 |
| PAR | 保留原有局部像素细化模块，默认 20 次、膨胀率 1/2/4/8/12/24；仅处理当前图像的正类通道 |
| 数据归一化 | 统一为 CLIP RGB mean/std；模型中完成归一化 |
| 数据读取 | 训练移除像素标注依赖，直接使用既有图像级标签字典；验证默认 VOC val 与 COCO val |
| 推理 | 直接使用训练后的分割解码器，无需逐图 LP 损失、类别标签或掩码区域再编码；尺度 0.75/1.0，翻转需显式启用 |

未保留无关的重复分割模型、离线固定模板 CAM 脚本、Grad-CAM 源码副本和未使用的 MMCV/MMSeg 组件，以免出现可被误用的旧训练路径。核心保留来源见根目录第三方说明。

## 七、COCO 掩码转换约定

可选的 `prepare_coco_masks.py` 读取官方实例标注；将排序后的 80 个类别 ID 映射到 1..80；大实例先绘制、小实例后覆盖；crowd 仅在没有有效实例标签的位置填为 255，其余未标注区为 0。输出文件使用官方完整图片 stem。

这是一种明确的实例到语义转换策略；论文没有给出该转换策略。因此它可能与原实验所用掩码在重叠、crowd 等像素上不同，不能混用掩码协议比较精确指标。已有基准掩码时不需要运行此脚本。

## 八、交付与未做事项

- 源码在新建的 LenCLIP 目录中修改，原始输入目录未被覆盖。
- 附带数据划分及图像级标签；未附带训练图像、预训练/训练权重、论文 PDF 或任何实验结果。
- 只做文件内容阅读、接口和梯度路径的静态审阅，以及命名、打包清单检查。
- 未运行 Python 源码导入、语法编译、单元测试、前向/反向、训练、推理、数据集评测或性能测试。
- 因缺少原始丢失代码与实验验证，不能保证与原论文实现逐项一致或取得论文指标。
