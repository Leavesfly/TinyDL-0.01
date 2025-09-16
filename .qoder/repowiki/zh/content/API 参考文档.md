# API 参考文档

<cite>
**本文档中引用的文件**  
- [Variable.java](file://src/main/java/io/leavesfly/tinydl/func/Variable.java)
- [Trainer.java](file://src/main/java/io/leavesfly/tinydl/mlearning/Trainer.java)
- [Model.java](file://src/main/java/io/leavesfly/tinydl/mlearning/Model.java)
- [Optimizer.java](file://src/main/java/io/leavesfly/tinydl/mlearning/optimize/Optimizer.java)
- [Loss.java](file://src/main/java/io/leavesfly/tinydl/mlearning/loss/Loss.java)
- [Block.java](file://src/main/java/io/leavesfly/tinydl/nnet/Block.java)
- [Layer.java](file://src/main/java/io/leavesfly/tinydl/nnet/Layer.java)
- [Parameter.java](file://src/main/java/io/leavesfly/tinydl/nnet/Parameter.java)
- [Function.java](file://src/main/java/io/leavesfly/tinydl/func/Function.java)
- [LayerAble.java](file://src/main/java/io/leavesfly/tinydl/nnet/LayerAble.java)
</cite>

## 目录
1. [简介](#简介)
2. [核心类与接口](#核心类与接口)
3. [Variable 类](#variable-类)
4. [Function 与数学运算](#function-与数学运算)
5. [神经网络层与块](#神经网络层与块)
6. [模型与训练](#模型与训练)
7. [优化器](#优化器)
8. [损失函数](#损失函数)

## 简介
本API参考文档旨在为TinyDL框架提供详尽的开发者参考。文档覆盖了框架中的核心类、接口、方法和字段，按功能模块组织，便于开发者快速查找和理解。所有内容均基于源码分析，确保准确性和实用性。

## 核心类与接口
本节概述框架中最关键的类和接口，它们构成了TinyDL的核心架构。

**Section sources**
- [Variable.java](file://src/main/java/io/leavesfly/tinydl/func/Variable.java#L1-L338)
- [Function.java](file://src/main/java/io/leavesfly/tinydl/func/Function.java#L1-L92)
- [LayerAble.java](file://src/main/java/io/leavesfly/tinydl/nnet/LayerAble.java#L1-L83)

## Variable 类
`Variable` 类是框架中所有数值数据的容器，代表计算图中的一个节点。它封装了数值（NdArray）、梯度以及生成它的函数（Function），是自动微分机制的基础。

### 构造方法
- `Variable(NdArray _value)`  
  使用指定的NdArray值构造一个变量。  
  **参数**: `_value` - 变量的初始值，不可为空。  
  **异常**: 若 `_value` 为 `null`，抛出 `RuntimeException`。

- `Variable(Number number)`  
  使用一个基本数值构造一个标量变量。  
  **参数**: `number` - 基本数值。  
  **异常**: 若 `number` 为 `null`，抛出 `RuntimeException`。

- `Variable(NdArray _value, String _name)`  
  构造一个带名称的变量。  
  **参数**: `_value` - 变量值；`_name` - 变量名称。

- `Variable(NdArray _value, String _name, boolean _requireGrad)`  
  构造一个可指定是否需要梯度的变量。  
  **参数**: `_value` - 变量值；`_name` - 变量名称；`_requireGrad` - 是否需要计算梯度。

### 核心方法
- `void backward()`  
  执行反向传播，计算从当前变量到其所有输入变量的梯度。  
  **说明**: 如果 `requireGrad` 为 `false`，则不进行计算。梯度初始化为1。  
  **异常**: 计算图不完整或梯度大小不匹配时抛出 `RuntimeException`。

- `void unChainBackward()`  
  切断计算图，用于RNN等场景以防止梯度爆炸。  
  **说明**: 递归地清除当前变量及其所有输入变量的 `creator` 引用。

- `void clearGrad()`  
  清除当前变量的梯度，将其设为 `null`。

- `Variable setRequireGrad(boolean _requireGrad)`  
  设置是否需要计算该变量的梯度。  
  **返回**: 返回当前变量实例，支持链式调用。

### 数学运算方法
所有数学运算方法都返回一个新的 `Variable` 实例，代表运算结果。

- `Variable add(Variable other)`  
  执行加法运算。  
  **返回**: `this + other`。

- `Variable sub(Variable other)`  
  执行减法运算。  
  **返回**: `this - other`。

- `Variable mul(Variable other)`  
  执行乘法运算。  
  **返回**: `this * other`。

- `Variable div(Variable other)`  
  执行除法运算。  
  **返回**: `this / other`。

- `Variable neg()`  
  执行取反运算。  
  **返回**: `-this`。

- `Variable squ()`  
  计算平方。  
  **返回**: `this^2`。

- `Variable pow(float pow)`  
  计算幂。  
  **参数**: `pow` - 幂指数。  
  **返回**: `this^pow`。

- `Variable exp()`  
  计算自然指数。  
  **返回**: `e^this`。

- `Variable sin()` / `Variable cos()` / `Variable tanh()` / `Variable log()`  
  分别计算正弦、余弦、双曲正切和自然对数。

- `Variable softMax()`  
  在最后一个轴上应用SoftMax函数。

- `Variable clip(float min, float max)`  
  将变量值裁剪到 `[min, max]` 区间。

- `Variable max(int _axis, boolean _keepdims)` / `Variable min(int _axis, boolean _keepdims)`  
  沿指定轴计算最大值/最小值。

### 张量操作方法
- `Variable broadcastTo(Shape shape)`  
  将变量广播到指定形状。

- `Variable matMul(Variable other)`  
  执行矩阵乘法。

- `Variable reshape(Shape shape)`  
  重塑变量的形状。

- `Variable sum()` / `Variable sumTo(Shape shape)`  
  计算所有元素的和，或将和广播到指定形状。

- `Variable transpose()`  
  转置变量。

- `Variable linear(Variable w, Variable b)`  
  执行线性变换 `y = xW + b`。如果 `b` 为 `null`，则只计算 `xW`。

- `Variable getItem(int[] _rowSlices, int[] _colSlices)`  
  获取指定切片的子张量。

### 损失函数方法
- `Variable meanSquaredError(Variable other)`  
  计算均方误差损失。  
  **返回**: `MeanSE` 函数的输出。

- `Variable softmaxCrossEntropy(Variable other)`  
  计算SoftMax交叉熵损失。  
  **返回**: `SoftmaxCE` 函数的输出。

**Section sources**
- [Variable.java](file://src/main/java/io/leavesfly/tinydl/func/Variable.java#L1-L338)

## Function 与数学运算
`Function` 是所有数学运算的抽象基类。每个具体的运算（如加法、Sigmoid）都是 `Function` 的一个子类。

### Function 抽象类
- `Variable call(Variable... _inputs)`  
  执行函数的前向传播，并构建计算图。  
  **参数**: `_inputs` - 输入变量数组。  
  **返回**: 输出变量。  
  **异常**: 输入数量不匹配时抛出 `RuntimeException`。  
  **说明**: 如果全局配置 `Config.train` 为 `true`，则会设置 `output.setCreator(this)` 以构建计算图。

- `abstract NdArray forward(NdArray... inputs)`  
  定义前向传播的具体计算逻辑。

- `abstract List<NdArray> backward(NdArray yGrad)`  
  定义反向传播的梯度计算逻辑。

- `abstract int requireInputNum()`  
  指定该函数所需的输入变量数量，-1 表示任意数量。

**Section sources**
- [Function.java](file://src/main/java/io/leavesfly/tinydl/func/Function.java#L1-L92)

## 神经网络层与块
### LayerAble 抽象类
`LayerAble` 是所有神经网络层和块的基类，继承自 `Function`。

- `abstract void init()`  
  初始化层的参数和内部状态。

- `abstract Variable layerForward(Variable... inputs)`  
  执行层的前向传播。

- `abstract void clearGrads()`  
  清除所有参数的梯度。

- `void addParam(String paramName, Parameter value)`  
  添加一个参数到 `params` 映射中，键为 `name.paramName`。

- `Parameter getParamBy(String paramName)`  
  根据名称获取参数。

### Layer 与 Block
- `Layer` 是 `LayerAble` 的直接子类，表示一个具体的层（如全连接层、卷积层）。
- `Block` 是一个复合结构，可以包含多个 `LayerAble`（层或其他块），用于构建复杂的网络。

- `Block.addLayer(LayerAble layerAble)`  
  向块中添加一个层或子块。

- `Block.layerForward(Variable... inputs)`  
  按顺序执行所有层的前向传播。

- `Block.getAllParams()`  
  递归获取该块及其所有子层/子块的所有参数。

- `Block.resetState()`  
  重置所有RNN层或子块的状态。

**Section sources**
- [LayerAble.java](file://src/main/java/io/leavesfly/tinydl/nnet/LayerAble.java#L1-L83)
- [Layer.java](file://src/main/java/io/leavesfly/tinydl/nnet/Layer.java#L1-L34)
- [Block.java](file://src/main/java/io/leavesfly/tinydl/nnet/Block.java#L1-L90)

## 模型与训练
### Model 类
`Model` 类封装了一个神经网络块（`Block`），并提供了训练和推理的接口。

- `Model(String _name, Block _block)`  
  构造一个模型。  
  **参数**: `_name` - 模型名称；`_block` - 网络结构块。

- `Variable forward(Variable... inputs)`  
  执行模型的前向传播。  
  **返回**: `block.layerForward(inputs)`。

- `void clearGrads()`  
  清除模型中所有参数的梯度。

- `Map<String, Parameter> getAllParams()`  
  获取模型中所有参数的映射。

- `void resetState()`  
  重置模型中所有RNN层的状态。

- `void plot()`  
  使用UML工具绘制当前计算图。

- `void save(File modelFile)` / `static Model load(File modelFile)`  
  序列化和反序列化模型。

**Section sources**
- [Model.java](file://src/main/java/io/leavesfly/tinydl/mlearning/Model.java#L1-L86)

### Trainer 类
`Trainer` 类负责模型的训练流程。

- `Trainer(int _maxEpoch, Monitor _monitor, Evaluator _evaluator)`  
  构造一个训练器。  
  **参数**: `_maxEpoch` - 最大训练轮数；`_monitor` - 监控器；`_evaluator` - 评估器。

- `void init(DataSet _dataSet, Model _model, Loss _loss, Optimizer _optimizer)`  
  初始化训练器。  
  **说明**: 准备数据集，设置模型、损失函数和优化器。

- `void train(boolean shuffleData)`  
  执行训练。  
  **流程**: 对每个epoch，遍历训练数据的每个batch，计算损失，反向传播，更新参数。  
  **参数**: `shuffleData` - 是否在每个epoch开始时打乱数据。

- `void evaluate()`  
  使用评估器对模型进行评估。

**Section sources**
- [Trainer.java](file://src/main/java/io/leavesfly/tinydl/mlearning/Trainer.java#L1-L106)

## 优化器
### Optimizer 抽象类
`Optimizer` 是所有优化算法的基类。

- `Optimizer(Model target)`  
  构造一个优化器。  
  **参数**: `target` - 需要优化的模型。

- `void update()`  
  更新模型中所有参数。  
  **说明**: 遍历 `target.getAllParams()` 并对每个参数调用 `updateOne`。

- `abstract void updateOne(Parameter parameter)`  
  定义如何更新单个参数的具体逻辑（如SGD、Adam）。

**Section sources**
- [Optimizer.java](file://src/main/java/io/leavesfly/tinydl/mlearning/optimize/Optimizer.java#L1-L28)

## 损失函数
### Loss 抽象类
`Loss` 是所有损失函数的基类。

- `abstract Variable loss(Variable y, Variable predict)`  
  计算真实值 `y` 和预测值 `predict` 之间的损失。  
  **返回**: 一个表示损失值的 `Variable`。

**Section sources**
- [Loss.java](file://src/main/java/io/leavesfly/tinydl/mlearning/loss/Loss.java#L1-L10)