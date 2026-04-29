# 机器学习实验：基于CNN的手写数字识别

## 1. 学生信息

- **姓名**：唐为波
- **学号**：112304260144
- **班级**：数据1231

---

## 2. 实验任务

本实验基于MNIST手写数字数据集，使用卷积神经网络（CNN）完成从模型训练到应用部署的完整流程。

本实验重点包括：
- 卷积神经网络（CNN）模型搭建
- 超参数调优（优化器、学习率、Batch Size）
- 数据增强
- 模型部署为Web应用
- Kaggle结果提交

---

## 3. 比赛与提交信息

**比赛名称**：Digit Recognizer

**比赛链接**：https://www.kaggle.com/competitions/digit-recognizer/overview

**提交日期**：2026-04-29

**GitHub 仓库地址**：https://github.com/wenchilail/tangweibo112304260144CNN

---

## 4. Kaggle成绩

请填写你最终提交到Kaggle的结果：

**Public Score**：待提交

**Private Score（如有）**：待提交

---

## 5. 实验方法说明

### （1）模型架构

请说明你使用的模型架构：

**我的做法**：

卷积神经网络（CNN）架构：
- 第1层卷积：Conv2d(1, 32, 3) + BatchNorm + MaxPool
- 第2层卷积：Conv2d(32, 64, 3) + BatchNorm + MaxPool
- Dropout：0.25
- 全连接1：Linear(64*7*7, 128) + ReLU
- 全连接2：Linear(128, 10)

### （2）超参数与训练

请说明你的训练设置：

**我的做法**：

- **优化器**：Adam
- **学习率**：0.001
- **Batch Size**：128
- **训练Epoch**：30（Early Stopping）
- **数据增强**：RandomRotation(10)
- **损失函数**：CrossEntropyLoss

### （3）实验对比

请说明你做的对比实验：

**我的做法**：

完成了4组对比实验：
- **Exp1**：SGD优化器，学习率0.01，Batch Size 64
- **Exp2**：Adam优化器，学习率0.001，Batch Size 64
- **Exp3**：Adam优化器，学习率0.001，Batch Size 128
- **Exp4**：Adam优化器，学习率0.001，Batch Size 64 + 数据增强

最终采用：**Exp4**（Adam + 数据增强 + Early Stopping）

---

## 6. 实验流程

请简要说明你的实验流程。

**我的实验流程**：

1. 读取训练集和测试集数据
2. 数据预处理（归一化）
3. 搭建CNN模型
4. 进行4组对比实验
5. 选择最佳超参数配置
6. 训练最终模型
7. 对测试集进行预测
8. 生成 submission 文件并提交到Kaggle
9. 部署Web应用（图片上传 + 手写画板）

---

## 7. 文件说明

请说明仓库中各文件或文件夹的作用。

**我的项目结构**：

```
10人工神经网络/
├── README.md                   # 项目说明文档
├── .gitignore                  # Git忽略文件
├── code/                       # 实验代码
│   ├── app.py                  # Flask Web应用
│   ├── train_mnist.py          # 基础模型训练代码
│   ├── train_mnist_final.py    # 最终模型训练代码
│   ├── train_mnist_optimized.py # 优化模型训练代码
│   ├── experiment.py           # 对比实验代码
│   ├── quick_experiment.py     # 快速对比实验
│   ├── check_torch.py          # PyTorch环境检查
│   └── requirements.txt        # Python依赖列表
├── report/                     # 实验报告
│   ├── CNN手写数字识别实验报告.md  # 完整实验报告
│   └── CNN手写数字识别实验模板.md  # 实验报告模板
└── results/                     # 实验结果
    ├── loss_curves.png         # 训练Loss曲线图
    ├── index.html              # HTML界面文件
    ├── templates/
    │   └── index.html          # Web应用界面
    └── digit-recognizer/
        └── submission_final.csv  # Kaggle提交文件
```

---

## 8. 使用说明

### （1）安装依赖

```bash
cd code
pip install -r requirements.txt
```

### （2）运行Web应用

```bash
cd code
python app.py
```

访问地址：http://localhost:5000

### （3）训练模型

```bash
cd code
python train_mnist_final.py
```

### （4）运行对比实验

```bash
cd code
python experiment.py
```
