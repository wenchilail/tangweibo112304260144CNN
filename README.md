# CNN手写数字识别实验

## 项目简介

本项目基于MNIST手写数字数据集，使用卷积神经网络（CNN）完成从模型训练到应用部署的完整流程。

## 项目结构

```
project/
├── app.py                  # Web应用入口
├── model.pth               # 训练好的模型权重（best_model_final.pth）
├── requirements.txt        # 依赖列表
├── README.md               # 项目说明
├── train_mnist.py          # 基础版本训练代码
├── train_mnist_final.py    # 最终版本训练代码
├── experiment.py           # 对比实验代码
├── digit-recognizer/       # 数据集目录
│   ├── train.csv
│   ├── test.csv
│   └── submission_final.csv
└── CNN手写数字识别实验模板.md   # 实验报告模板
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行Web应用

```bash
python app.py
```

应用启动后会自动在浏览器中打开，你可以通过以下两种方式使用：
- **图片上传**：上传一张手写数字图片进行识别
- **手写画板**：在网页上直接手写数字进行识别

### 3. 训练模型

如果需要重新训练模型：

```bash
python train_mnist_final.py
```

## 模型性能

- **最佳验证准确率**: 99.27%
- **Kaggle Score**: 待提交
- **模型架构**: 2层CNN + 2层FC

## 对比实验

运行对比实验代码：

```bash
python experiment.py
```

这将完成4组对比实验并生成Loss曲线图。

## 技术栈

- **深度学习框架**: PyTorch
- **Web框架**: Gradio
- **数据处理**: pandas, numpy
- **可视化**: matplotlib

## 作者

- 姓名: 待填写
- 学号: 待填写
- 班级: 待填写
