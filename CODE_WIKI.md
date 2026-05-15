# RadarMOTR Code Wiki

## 项目概述

**RadarMOTR** 是一个基于 Transformer 神经网络的多目标跟踪系统，专门用于处理 Range-Doppler Maps（距离-多普勒图）。该项目基于 [MOTRv2](https://github.com/megvii-research/MOTRv2) 和 [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) 构建。

### 核心特性

- **基于 Transformer 的跟踪**: 使用神经网络替代传统 Kalman Filter 进行目标关联
- **支持多种跟踪器**: 集成 RadarMOTR、SORT、Kalman Filter 三种跟踪算法
- **多后端支持**: 支持 ResNet18 和 ResNet50 骨干网络
- **分布式训练**: 支持多 GPU 数据并行训练

### 项目来源

- 发表论文: 2024 International Radar Conference (RADAR)
- 预印本: [Accepted Preprint](https://bwsyncandshare.kit.edu/s/zCgc5o89L44oN5a)
- 数据集: [RADTrack](https://github.com/madeit07/RADTrack)

---

## 项目架构

```
RadarMOTR/
├── configs/                    # 配置文件目录
│   ├── base/                   # 基础配置
│   │   ├── dataset.yaml        # 数据集配置
│   │   ├── model.yaml          # 模型配置
│   │   └── misc.yaml           # 杂项配置
│   ├── trackers/               # 跟踪器配置
│   │   ├── kf.yaml             # Kalman Filter 配置
│   │   ├── radarmotr.yaml      # RadarMOTR 配置
│   │   └── sort.yaml           # SORT 配置
│   ├── train.yaml              # 训练配置
│   ├── eval.yaml               # 评估配置
│   ├── resnet18.yaml           # ResNet18 骨干网络配置
│   └── resnet50.yaml           # ResNet50 骨干网络配置
├── datasets/                   # 数据处理模块
│   ├── radartrack.py           # 雷达数据集加载器
│   ├── transforms.py           # 数据增强变换
│   ├── radar_transforms.py     # 雷达专用数据增强
│   ├── data_prefetcher.py      # 数据预取器
│   └── samplers.py             # 分布式采样器
├── models/                     # 核心模型模块
│   ├── radarmotr.py            # RadarMOTR 主模型
│   ├── deformable_transformer_plus.py  # 可变形Transformer
│   ├── backbones/              # 骨干网络
│   │   ├── backbone.py         # 骨干网络基类
│   │   └── circular_resnet.py  # 循环ResNet
│   ├── structures/             # 数据结构
│   │   ├── instances.py        # Instance 数据结构
│   │   └── boxes.py           # Bounding Box 工具
│   ├── ops/                    # 自定义算子
│   │   └── ms_deform_attn.py  # 多尺度可变形注意力
│   ├── clip_matcher.py         # Clip 匹配器
│   ├── matcher.py              # Hungarian 匹配器
│   ├── position_encoding.py    # 位置编码
│   ├── optimizer.py            # 优化器构建
│   └── tracker.py              # 跟踪器基类
├── trackers/                   # 跟踪器实现
│   ├── tracker_base.py         # 跟踪器基类
│   ├── radarmotr.py            # RadarMOTR 跟踪器
│   ├── sort.py                 # SORT 跟踪器
│   ├── kf.py                   # Kalman Filter 跟踪器
│   └── util/                   # 跟踪器工具
│       ├── sort.py
│       └── kf.py
├── util/                       # 工具函数
│   ├── misc.py                 # 分布式训练辅助函数
│   ├── box_ops.py              # 边界框操作
│   ├── checkpoint.py           # 检查点管理
│   └── tool.py                 # 模型加载工具
├── tools/                      # 辅助工具
│   ├── visualize.py            # 可视化工具
│   └── eval_to_markdown.py     # 评估结果转换
├── main.py                     # 训练入口
├── eval.py                     # 评估入口
└── engine.py                   # 训练/评估引擎
```

---

## 核心模块详解

### 1. 模型模块 (`models/`)

#### 1.1 RadarMOTR 主模型 (`models/radarmotr.py`)

**RadarMOTR** 是项目的核心模型，集成了 Transformer 架构进行多目标跟踪。

**主要类**: `RadarMOTR(nn.Module)`

**初始化参数**:
- `backbone`: 骨干网络
- `transformer`: Transformer 编码器-解码器
- `num_classes`: 目标类别数
- `num_queries`: 查询槽数量
- `num_feature_levels`: 特征层级数
- `criterion`: 损失函数计算器
- `qim`: Query 交互模块
- `post_process`: 后处理模块
- `track_base`: 运行时跟踪基类
- `aux_loss`: 是否使用辅助损失
- `with_box_refine`: 是否使用边界框迭代细化
- `two_stage`: 是否使用两阶段检测
- `query_denoise`: 查询去噪参数

**核心方法**:

| 方法 | 说明 |
|------|------|
| `forward(data)` | 训练时的前向传播，处理连续帧 |
| `predict(img, ori_img_size, track_instances, proposals)` | 推理时的预测方法 |
| `_generate_empty_tracks()` | 生成空的跟踪实例 |
| `_forward_backbone(samples)` | 骨干网络前向传播 |
| `_forward_single_image(samples, track_instances, gtboxes)` | 单帧前向传播 |
| `_post_process_single_image(pred, track_instances, is_last, gt_instances)` | 单帧后处理 |

**模型构建函数**: `build(args)` - 根据配置构建完整的 RadarMOTR 模型

#### 1.2 可变形 Transformer (`models/deformable_transformer_plus.py`)

**核心类**: `DeformableTransformer`

**架构组成**:
- **Encoder**: `DeformableTransformerEncoder` - 多层可变形注意力编码器
- **Decoder**: `DeformableTransformerDecoder` - 多层可变形注意力解码器
- **Layer**: `DeformableTransformerEncoderLayer` / `DeformableTransformerDecoderLayer`

**关键特性**:
- 支持多尺度特征融合
- 可配置编码器/解码器层数
- 支持 memory bank 用于时序建模
- 支持 self-cross attention 机制

#### 1.3 骨干网络 (`models/backbones/`)

**支持的后端网络**:
- `circular_resnet18`: 循环填充的 ResNet18
- `resnet18`: 标准 ResNet18
- `resnet50`: 标准 ResNet50

**特征输出**:
- 多层级特征: layer2, layer3, layer4
- 输出通道: [128, 256, 512] (ResNet18) / [512, 1024, 2048] (ResNet50)
- 步长: [8, 16, 32]

#### 1.4 Query 交互模块 (`models/qim.py`)

**核心类**: `QueryInteractionModuleV2`

**功能**: 在帧间传递和更新跟踪查询嵌入

**关键方法**:
- `_select_active_tracks()`: 选择活跃的跟踪
- `_update_track_embedding()`: 更新跟踪嵌入

#### 1.5 匹配器 (`models/matcher.py`, `models/clip_matcher.py`)

**HungarianMatcher**: 使用 Hungarian 算法进行二分图匹配

**ClipMatcher**: 扩展匹配器，支持 clip 级别的损失计算

**匹配成本**:
- 分类成本 (Focal Loss)
- 边界框 L1 成本
- GIoU 成本

### 2. 数据处理模块 (`datasets/`)

#### 2.1 雷达数据集 (`datasets/radartrack.py`)

**核心类**:

| 类名 | 说明 |
|------|------|
| `RadarTrack` | 主数据集类 |
| `RadarSequence` | 单个雷达序列 |
| `RadarTrackSequences` | 验证集数据集 |

**数据格式**:
- 支持格式: RDTrack, RATrack
- 标注格式: MOT 格式
- 检测格式: 外部检测器输出

**数据增强**:
- `MotTranslateBoxes`: 边界框平移
- `MotRandomReverseAndHFlip`: 随机翻转
- `MotRandomNoise`: 随机噪声
- `MotNormalize`: 标准化

### 3. 跟踪器模块 (`trackers/`)

#### 3.1 跟踪器基类 (`trackers/tracker_base.py`)

**核心类**: `Tracker` (抽象基类)

**接口方法**:
- `track_frame(frame, data)`: 单帧跟踪
- `track(loader, seq_id, output_dir)`: 批量跟踪
- `reset()`: 重置跟踪状态

#### 3.2 跟踪器实现

| 跟踪器 | 文件 | 说明 |
|--------|------|------|
| RadarMOTR | `radarmotr.py` | 基于神经网络的跟踪 |
| SORT | `sort.py` | 基于 IOU 的跟踪 |
| Kalman Filter | `kf.py` | 基于卡尔曼滤波的跟踪 |

### 4. 工具模块 (`util/`)

#### 4.1 分布式训练工具 (`util/misc.py`)

**关键类和函数**:

| 函数/类 | 说明 |
|---------|------|
| `MetricLogger` | 训练指标记录器 |
| `SmoothedValue` | 平滑值追踪 |
| `NestedTensor` | 嵌套张量容器 |
| `init_distributed_mode()` | 初始化分布式训练 |
| `nested_tensor_from_tensors()` | 张量转嵌套张量 |
| `reduce_dict()` / `reduce_dict_async()` | 分布式规约 |

#### 4.2 边界框操作 (`util/box_ops.py`)

- `box_giou()`: 计算 GIoU
- `box_cxcywh_to_xyxy()`: 格式转换
- `box_xyxy_to_cxcywh()`: 格式转换

---

## 关键数据结构

### Instances (`models/structures/instances.py`)

管理图像中的实例集合，支持动态字段。

**主要属性**:
- `boxes`: 边界框 [N, 4]
- `labels`: 类别标签 [N]
- `scores`: 置信度分数 [N]
- `obj_ids`: 目标 ID [N]
- `matched_gt_idxes`: 匹配的 GT 索引 [N]
- `pred_logits`: 预测类别 logits [N, num_classes]
- `pred_boxes`: 预测边界框 [N, 4]
- `output_embedding`: 输出嵌入 [N, hidden_dim]
- `mem_bank`: 记忆库 [N, max_his_length, hidden_dim]

**支持操作**:
- `cat()`: 合并多个 Instances
- `to()`: 设备转换
- `__getitem__()`: 索引和切片

### Boxes (`models/structures/boxes.py`)

边界框数据结构，支持多种格式。

**支持格式**:
- `xyxy`: [x1, y1, x2, y2]
- `xywh`: [x, y, w, h]
- `cxcywh`: [cx, cy, w, h]

---

## 配置文件说明

### 训练配置 (`configs/train.yaml`)

```yaml
start_epoch: 0
epochs: 80
lr_drop: 40
output_dir: 'data/RadarMOTR'
```

### 模型配置 (`configs/base/model.yaml`)

```yaml
# 学习率
lr: 0.0002
lr_backbone: 0.00002

# Transformer 配置
hidden_dim: 256
nheads: 8
enc_layers: 6
dec_layers: 6
dim_feedforward: 1024

# 骨干网络
backbone: 'resnet50'
num_feature_levels: 4

# 跟踪配置
num_queries: 10
score_threshold: 0.6
filter_score_threshold: 0.5
miss_tolerance: 10
```

### 评估配置 (`configs/trackers/radarmotr.yaml`)

```yaml
tracker: 'RadarMOTR'
model_path: 'data/models/radarmotr_r18.pth'
score_threshold: 0.5
filter_score_threshold: 0.5
miss_tolerance: 5
area_threshold: 0
```

---

## 依赖关系

### 核心依赖

```
torch>=2.1.0
torchvision>=0.16.0
```

### 项目依赖

```
scipy==1.11.4          # 优化算法
sacred==0.8.5          # 实验管理
PyYAML==6.0.1          # 配置解析
Pillow==10.1.0         # 图像处理
tqdm==4.66.1           # 进度条
pandas==2.1.4          # 数据处理
opencv-python-headless # 图像处理
```

### 评估依赖

```
numpy==1.23.5
scipy==1.11.4
pycocotools==2.0.6
matplotlib==3.8.2
tabulate==0.9.0
filterpy==1.4.5        # Kalman Filter
scikit-image==0.22.0
```

### CUDA 依赖

```
MultiScaleDeformableAttention (自定义算子)
- 需要 GNU G++ Compiler <11
- 需要 CUDA Toolkit 11.8
```

---

## 运行方式

### 环境安装

```bash
# 1. 克隆仓库
git clone --recurse-submodules https://github.com/madeit07/RadarMOTR.git
cd RadarMOTR

# 2. 创建 conda 环境
conda create -n radarmotr python=3.11
conda activate radarmotr

# 3. 安装 PyTorch
conda install pytorch=2.1 torchvision=0.16 pytorch-cuda=11.8 -c pytorch -c nvidia

# 4. 安装 CUDA Toolkit
conda install cuda-libraries-dev cuda-nvcc cuda-nvtx cuda-cupti -c nvidia/label/cuda-11.8.0

# 5. 安装其他依赖
pip install -r requirements.txt

# 6. 编译 MultiScaleDeformableAttention
cd ./models/ops
./make.sh
```

### 训练

```bash
# 单 GPU 训练
python main.py with resnet18

# 多 GPU 训练
torchrun --standalone --nnodes 1 --nproc_per_node 4 --max_restarts 2 main.py with resnet18

# 使用预训练权重
python main.py with resnet18 pretrained=<PATH_TO_PRETRAINED>/motrv2_base_r18_v2.pth
```

### 评估

```bash
# RadarMOTR 跟踪器评估
python eval.py with radarmotr resnet18 split=val model_path=<PATH>/radarmotr_r18.pth

# Kalman Filter 评估
python eval.py with kf split=val tracker_dirname=kf

# SORT 评估
python eval.py with sort split=val tracker_dirname=sort
```

### 自定义配置

```bash
# 修改学习率
python main.py with resnet18 lr=0.0001

# 修改批量大小
python main.py with resnet18 batch_size=4

# 启用调试模式
python main.py with resnet18 debug
```

---

## 数据集准备

### RDTrack 数据集

1. 从 [Google Drive](https://drive.google.com/drive/folders/1h0Tv5X86o7G_yNxn9_nTYm_3EKvdRcIO) 下载
2. 解压并保存到 `data/dataset/RDTrack`

### 数据格式

```
RDTrack/
├── rdtrack-train/
│   └── seq_001/
│       ├── seqinfo.ini
│       ├── img1/
│       │   └── 000001.png
│       ├── det/
│       │   └── det.txt
│       └── gt/
│           └── gt.txt
├── rdtrack-val/
└── seqmaps/
    └── rdtrack-train.txt
```

### 预训练权重

| 权重 | 骨干网络 | 路径 |
|------|----------|------|
| motrv2_base_r18_v2.pth | ResNet18 | data/models/ |
| motrv2_base_v2.pth | ResNet50 | data/models/ |
| radarmotr_r18.pth | ResNet18 | data/models/ |

---

## 训练流程

### 主训练循环 (`main.py`)

```
1. 初始化分布式环境
   ↓
2. 构建模型和损失函数
   ↓
3. 构建数据集和数据加载器
   ↓
4. 构建优化器和学习率调度器
   ↓
5. 加载预训练权重（如有）
   ↓
6. 训练循环:
   ├── 前向传播
   ├── 计算损失
   ├── 反向传播
   ├── 梯度裁剪
   ├── 参数更新
   └── 学习率调整
   ↓
7. 验证循环（如配置启用）
   ↓
8. 保存检查点
```

### 模型前向传播 (`RadarMOTR.forward`)

```
输入: 连续帧数据
  ↓
对每帧进行:
  ├── 骨干网络特征提取
  ├── Transformer 编码器处理
  ├── Transformer 解码器处理
  ├── 分类和边界框预测
  ├── 跟踪匹配
  └── Query 交互更新
  ↓
输出: 预测结果 + 损失
```

---

## 评估指标

### 支持的评估指标

| 指标类别 | 具体指标 |
|----------|----------|
| HOTA | HOTA, DetA, AssA |
| CLEAR | MOTA, MOTP, IDF1 |
| Identity | IDs, FG, GT |
| 计数 | FP, FN, Frag |

### 评估输出

评估结果保存到 `data/trackers/<dataset>-<split>/<tracker_name>/`

---

## 扩展指南

### 添加新的跟踪器

1. 继承 `Tracker` 基类
2. 实现 `track_frame()` 方法
3. 在 `eval.py` 的 `build_tracker()` 函数中注册

### 添加新的骨干网络

1. 在 `models/backbones/` 中实现
2. 在 `BACKBONES` 字典中注册
3. 更新配置文件的 `backbone` 参数

### 修改损失函数

1. 修改 `models/clip_matcher.py` 中的损失计算
2. 或创建新的 Criterion 类
3. 在 `models/radarmotr.py` 的 `build()` 函数中集成

---

## 常见问题

### Q: 训练时显存不足怎么办？

A: 尝试以下方法：
- 减小 `batch_size`
- 启用梯度检查点: `use_grad_checkpointing=true`
- 减少 `num_queries`
- 使用 ResNet18 骨干网络

### Q: 如何加速推理？

A: 建议：
- 使用较短的 `miss_tolerance`
- 调整 `score_threshold` 和 `filter_score_threshold`
- 使用批量推理

### Q: 模型无法收敛怎么办？

A: 检查：
- 学习率设置
- 数据增强配置
- 预训练权重是否正确加载
- `query_denoise` 参数

---

## 参考资料

- [MOTRv2](https://github.com/megvii-research/MOTRv2)
- [MOTR](https://github.com/megvii-research/MOTR)
- [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)
- [DETR](https://github.com/facebookresearch/detr)
- [RADTrack Dataset](https://github.com/madeit07/RADTrack)
