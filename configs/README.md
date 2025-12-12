# YAML配置文件使用指南

## 概述

训练脚本现在支持从YAML配置文件加载参数，提供三层配置优先级：

```
命令行参数 > YAML配置 > 代码默认值
```

## 快速开始

### 方法1：使用默认配置文件
```bash
python scripts/train.py
# 自动加载 configs/train.yaml
```

### 方法2：指定配置文件
```bash
python scripts/train.py --config configs/train_fast_test.yaml
```

### 方法3：YAML + 命令行覆盖
```bash
# 使用配置文件，但覆盖学习率
python scripts/train.py --config configs/train.yaml --learning-rate 5e-4

# 使用配置文件，但修改环境数
python scripts/train.py --config configs/train_stable.yaml --num-envs 2048
```
