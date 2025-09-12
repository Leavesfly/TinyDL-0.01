## TinyDL - 轻量级深度学习框架

![TinyDL架构图](img.png)

## 📖 项目简介

TinyDL 是一个用 Java 实现的轻量级深度学习框架，旨在提供简洁、清晰的深度学习核心功能实现。该框架参考了 PyTorch 的设计理念，实现了自动微分、神经网络层、优化器等核心组件，适合学习深度学习原理和进行小规模实验。

## ✨ 主要特性

- **🔢 多维数组支持**: 核心 `NdArray` 类支持标量、向量、矩阵等多维数据操作
- **🔄 自动微分**: 基于计算图的自动梯度计算，支持反向传播
- **🧠 神经网络层**: 实现了全连接层、卷积层、RNN层、LSTM层等常用网络层
- **📊 机器学习组件**: 包含数据集、损失函数、优化器、训练器等完整的机器学习工具链
- **📈 可视化支持**: 集成 JFreeChart 提供训练过程可视化
- **🎯 丰富示例**: 提供分类、回归、序列预测等多种应用示例

## 🏗️ 架构设计

TinyDL 秉承简洁分层清晰的原则，整体架构如下：

### 核心模块

1. **📦 ndarr包**: 核心类 `NdArray`，底层线性代数的简单实现，目前只实现CPU版本
2. **⚡ func包**: 核心类 `Function` 与 `Variable`，分别是抽象的数学函数与变量的抽象，用于在前向传播时自动构建计算图，实现自动微分功能
3. **🔗 nnet包**: 核心类 `Layer` 与 `Block` 表示神经网络的层和块，任何复杂的深度网络都是依赖这些Layer与Block的堆叠而成
4. **🎓 mlearning包**: 机器学习的通用组件，包括数据集、损失函数、优化算法、训练器、推理器、效果评估器等
5. **🎯 modality包**: 应用层范畴，目前深度学习主要应用于计算机视觉、自然语言处理以及强化学习三部分
6. **💡 example包**: 一些简单的能跑通的例子，包括机器学习的分类和回归问题

## 🚀 快速开始

### 环境要求

- Java 8+
- Maven 3.6+

### 依赖配置

```xml
<dependencies>
    <dependency>
        <groupId>jfree</groupId>
        <artifactId>jfreechart</artifactId>
        <version>1.0.7</version>
    </dependency>
    <dependency>
        <groupId>junit</groupId>
        <artifactId>junit</artifactId>
        <version>4.13.2</version>
        <scope>test</scope>
    </dependency>
</dependencies>
```

### 基本使用示例

#### 1. 创建变量和基本运算

```java
import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;

// 创建变量
Variable x = new Variable(new NdArray(2.0f)).setName("x");
Variable y = new Variable(new NdArray(3.0f)).setName("y");

// 基本运算
Variable z = x.add(y).mul(x); // z = (x + y) * x

// 反向传播计算梯度
z.backward();
System.out.println("x的梯度: " + x.getGrad()); // 输出: 5.0
```

#### 2. 构建简单神经网络

```java
import io.leavesfly.tinydl.nnet.block.MlpBlock;
import io.leavesfly.tinydl.mlearning.Model;

// 创建多层感知机
int inputSize = 2;
int hiddenSize = 10;
int outputSize = 1;
int batchSize = 32;

MlpBlock mlpBlock = new MlpBlock("MLP", batchSize, null, 
                                inputSize, hiddenSize, outputSize);
Model model = new Model("SimpleModel", mlpBlock);

// 前向传播
Variable input = new Variable(NdArray.likeRandom(-1, 1, new Shape(batchSize, inputSize)));
Variable output = model.forward(input);
```

## 📚 API 文档

### 核心类说明

#### NdArray
多维数组类，支持各种数学运算：
- `NdArray(float value)`: 创建标量
- `NdArray(float[][] data)`: 创建二维矩阵
- `add()`, `sub()`, `mul()`, `div()`: 基本数学运算
- `matMul()`: 矩阵乘法
- `reshape()`: 改变形状

#### Variable
变量类，支持自动微分：
- `setRequireGrad(boolean)`: 设置是否需要梯度
- `backward()`: 反向传播
- `clearGrad()`: 清除梯度
- 支持各种数学运算符重载

#### Layer & Block
神经网络层和块：
- `LinearLayer`: 全连接层
- `ConvLayer`: 卷积层
- `LstmLayer`: LSTM层
- `MlpBlock`: 多层感知机块

## 🎯 示例项目

### 1. 螺旋数据分类
```bash
# 运行螺旋数据分类示例
java -cp target/classes io.leavesfly.tinydl.example.classify.SpiralMlpExam
```

### 2. 手写数字识别
```bash
# 运行MNIST手写数字识别
java -cp target/classes io.leavesfly.tinydl.example.classify.MnistMlpExam
```

### 3. 曲线拟合
```bash
# 运行Sin曲线拟合
java -cp target/classes io.leavesfly.tinydl.example.regress.MlpSinExam
```

### 4. RNN序列预测
```bash
# 运行RNN余弦序列预测
java -cp target/classes io.leavesfly.tinydl.example.regress.RnnCosExam
```

## 🛠️ 开发计划

### TinyDL 0.02 版本计划

- [x] **完成NdArray的维度扩张，支持更高维度** (3.18-3.24)
- [x] **完善CNN层的支持和demo** (3.25-3.31)
- [x] **完善RNN层的支持和demo** (4.1-4.7)
- [ ] **语言模型的支持之wordVec** (4.8-4.14)
- [ ] **语言模型的支持之attention** (4.15-4.21)
- [ ] **Transformer的支持** (4.22-4.28)
- [ ] **GPT-2的支持和demo** (4.29-5.5)
- [ ] **训练效率的优化支持并行训练** (5.6-5.12)

## 🏃‍♂️ 编译和运行

```bash
# 编译项目
mvn clean compile

# 运行测试
mvn test

# 打包
mvn package
```

## 📁 项目结构

```
src/main/java/io/leavesfly/tinydl/
├── ndarr/          # 多维数组核心实现
├── func/           # 函数和变量抽象
├── nnet/           # 神经网络层和块
├── mlearning/      # 机器学习通用组件
├── modality/       # 应用领域相关
├── example/        # 示例代码
└── utils/          # 工具类
```

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 联系方式

- 项目维护者: [项目作者]
- 邮箱: [联系邮箱]
- 项目链接: [GitHub仓库地址]

---

**注意**: TinyDL 目前处于开发阶段，主要用于教学和研究目的。生产环境请使用成熟的深度学习框架如 PyTorch、TensorFlow 等。