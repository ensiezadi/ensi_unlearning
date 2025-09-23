# PLT_NET_Mini Machine Unlearning

本项目已集成PLT_NET_Mini植物数据集，用于进行机器遗忘实验。

## 数据集信息

- **数据集名称**: PLT_NET_Mini
- **来源**: OpenML Dataset ID 44293
- **类别数量**: 25个植物类别
- **样本数量**: 约12,000张图片

## 遗忘类别

随机选择的3个类别进行遗忘训练：
- `Cirsium vulgare` (Spear thistle - 矛蓟)
- `Fragaria vesca` (Wild strawberry - 野草莓)
- `Hypericum perforatum` (St. John's wort - 圣约翰草)

## 快速开始

### 1. 运行实验
```bash
# Windows
run_plt_net_mini.bat

# Linux/Mac
./run_plt_net_mini.sh
```

### 2. 手动运行
```bash
python main.py --run_ds PLTNetMini --backbone_arch RN50 --output_dir results/plt_net_mini_experiment
```

### 3. 测试数据集
在Jupyter中运行 `test copy.ipynb` 来验证数据集设置。

## 项目结构

```
├── datasets/plt_net_mini.py          # PLT_NET_Mini数据集类
├── configs/datasets/plt_net_mini.yaml # 数据集配置
├── forget_cls.py                     # 遗忘类别配置
├── main.py                          # 主实验脚本
├── test copy.ipynb                  # 数据集测试notebook
├── run_plt_net_mini.bat            # Windows运行脚本
└── run_plt_net_mini.sh             # Linux/Mac运行脚本
```

## 实验结果

实验完成后，结果将保存在 `results/plt_net_mini_experiment/` 目录中，包括：
- 遗忘效果评估
- 各类别性能变化
- 模型权重保存

## 注意事项

1. 首次运行会自动从OpenML下载数据集
2. 需要CUDA支持的GPU进行训练
3. 确保有足够的磁盘空间存储数据集和结果