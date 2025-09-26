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
- [NdArray.java](file://src/main/java/io/leavesfly/tinydl/ndarr/NdArray.java) - *在最近提交中更新*
- [MoEGPTModel.java](file://src/main/java/io/leavesfly/tinydl/modality/nlp/MoEGPTModel.java) - *新增于最近提交*
- [MoETransformerBlock.java](file://src/main/java/io/leavesfly/tinydl/modality/nlp/block/MoETransformerBlock.java) - *新增于最近提交*
- [Agent.java](file://src/main/java/io/leavesfly/tinydl/modality/rl/Agent.java) - *新增于最近提交*
- [BanditAgent.java](file://src/main/java/io/leavesfly/tinydl/modality/rl/agent/BanditAgent.java) - *新增于最近提交*
</cite>

## 更新摘要
**已更改内容**   
- 更新了 **NdArray 类** 部分，以反映其API变更，包括新的构造方法、静态工厂方法和操作。
- 新增了 **MoE相关API** 部分，详细介绍了 `MoEGPTModel` 和 `MoETransformerBlock` 类。
- 新增了 **强化学习相关API** 部分，涵盖了 `Agent` 和 `BanditAgent` 抽象类。

**新增部分**
- MoE相关API
- 强化学习相关API

**已弃用/移除部分**
- 无

**源码追踪系统更新**
- 添加了对新文件和修改文件的引用，确保所有引用的文件都在文档中有所提及。

## 目录
1. [简介](#简介)
2. [核心类与接口](#核心类与接口)
3. [Variable 类](#variable-类)
4. [Function 与数学运算](#function-与数学运算)
5. [神经网络层与块](#神经网络层与块)
6. [模型与训练](#模型与训练)
7. [优化器](#优化器)
8. [损失函数](#损失函数)
9. [NdArray 类](#ndarray-类)
10. [MoE相关API](#moe相关api)
11. [强化学习相关API](#强化学习相关api)

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

## NdArray 类
`NdArray` 类是框架中N维数组的核心实现，支持标量、向量、矩阵等多维数据结构。它是深度学习框架中高效内存管理和数学运算的基础。

### 构造方法
- `NdArray()`  
  默认构造方法，创建空的NdArray实例。  
  **注意**: 此构造方法不会初始化shape和buffer，需要手动设置。

- `NdArray(Number number)`  
  从标量值创建NdArray。  
  **参数**: `number` - 标量值。

- `NdArray(float[] data, Shape shape)`  
  从一维数据数组和形状创建NdArray。  
  **参数**: `data` - 一维数据数组；`shape` - 数组形状。  
  **异常**: 当数据长度与形状大小不匹配时抛出 `IllegalArgumentException`。

- `NdArray(float[] data)`  
  从一维数组创建NdArray，默认形状为(1, data.length)。  
  **参数**: `data` - 一维数据数组。

- `NdArray(Object data)`  
  从多维数组对象创建NdArray，支持2D、3D、4D数组的创建。  
  **参数**: `data` - 多维数组对象（float[][]、float[][][]或float[][][][]）。  
  **异常**: 当输入类型不支持时抛出 `IllegalArgumentException`。

- `NdArray(Shape shape)`  
  从指定形状创建空的NdArray，所有元素初始化为0。  
  **参数**: `shape` - 数组形状。

### 静态工厂方法
- `static NdArray zeros(Shape shape)`  
  创建指定形状的全零数组。  
  **参数**: `shape` - 数组形状。  
  **返回**: 全零数组。

- `static NdArray ones(Shape shape)`  
  创建指定形状的全一数组。  
  **参数**: `shape` - 数组形状。  
  **返回**: 全一数组。

- `static NdArray eye(Shape shape)`  
  创建指定形状的单位矩阵（对角矩阵）。  
  **参数**: `shape` - 矩阵形状（必须为方形矩阵）。  
  **返回**: 单位矩阵。  
  **异常**: 当形状不是矩阵或不是方形矩阵时抛出 `IllegalArgumentException`。

- `static NdArray like(Shape shape, Number value)`  
  创建指定形状和值的数组。  
  **参数**: `shape` - 数组形状；`value` - 填充值。  
  **返回**: 指定值填充的数组。

- `NdArray like(Number value)`  
  创建与当前数组形状相同但指定值的数组。  
  **参数**: `value` - 填充值。  
  **返回**: 指定值填充的数组。

- `static NdArray likeRandomN(Shape shape)` / `static NdArray likeRandomN(Shape shape, long seed)`  
  创建标准正态分布（均值为0，标准差为1）的随机数组。  
  **参数**: `shape` - 数组形状；`seed` - 随机种子，0表示使用默认种子。  
  **返回**: 标准正态分布随机数组。

- `static NdArray likeRandom(float min, float max, Shape shape)` / `static NdArray likeRandom(float min, float max, Shape shape, long seed)`  
  创建指定范围内的均匀分布随机数组。  
  **参数**: `min` - 最小值（包含）；`max` - 最大值（包含）；`shape` - 数组形状；`seed` - 随机种子，0表示使用默认种子。  
  **返回**: 均匀分布随机数组。

- `static NdArray linSpace(float min, float max, int num)`  
  创建线性空间数组（等间距排序数组）。  
  **参数**: `min` - 起始值；`max` - 结束值；`num` - 元素数量。  
  **返回**: 线性空间数组。  
  **异常**: 当数量小于等于0时抛出 `IllegalArgumentException`。

### 基础四则运算
- `NdArray add(NdArray other)`  
  数组加法运算，对应元素相加。  
  **参数**: `other` - 另一个操作数数组。  
  **返回**: 加法运算结果。  
  **异常**: 当两个数组形状不一致时抛出 `IllegalArgumentException`。

- `NdArray sub(NdArray other)`  
  数组减法运算，对应元素相减。  
  **参数**: `other` - 另一个操作数数组。  
  **返回**: 减法运算结果。  
  **异常**: 当两个数组形状不一致时抛出 `IllegalArgumentException`。

- `NdArray mul(NdArray other)`  
  数组乘法运算，对应元素相乘。  
  **参数**: `other` - 另一个操作数数组。  
  **返回**: 乘法运算结果。  
  **异常**: 当两个数组形状不一致时抛出 `IllegalArgumentException`。

- `NdArray mulNum(Number number)`  
  数组与标量相乘。  
  **参数**: `number` - 标量值。  
  **返回**: 乘法运算结果。

- `NdArray div(NdArray other)`  
  数组除法运算，对应元素相除。  
  **参数**: `other` - 另一个操作数数组。  
  **返回**: 除法运算结果。  
  **异常**: 当两个数组形状不一致时抛出 `IllegalArgumentException`；当除数接近0时抛出 `ArithmeticException`。

- `NdArray divNum(Number number)`  
  数组与标量相除。  
  **参数**: `number` - 标量值。  
  **返回**: 除法运算结果。  
  **异常**: 当除数为0时抛出 `ArithmeticException`。

### 逻辑运算
- `NdArray neg()`  
  取反操作，对数组每个元素取负值。  
  **返回**: 取反后的数组。

- `NdArray abs()`  
  绝对值运算，对数组每个元素取绝对值。  
  **返回**: 绝对值数组。

- `NdArray eq(NdArray other)`  
  相等比较运算，比较两个数组对应元素是否相等。  
  **参数**: `other` - 另一个操作数数组。  
  **返回**: 比较结果数组，1.0表示true，0.0表示false。  
  **异常**: 当两个数组形状不一致时抛出 `IllegalArgumentException`。

- `NdArray gt(NdArray other)`  
  大于比较运算，比较当前数组元素是否大于另一个数组对应元素。  
  **参数**: `other` - 另一个操作数数组。  
  **返回**: 比较结果数组，1.0表示大于，0.0表示不大于。  
  **异常**: 当两个数组形状不一致时抛出 `IllegalArgumentException`。

- `NdArray lt(NdArray other)`  
  小于比较运算，比较当前数组元素是否小于另一个数组对应元素。  
  **参数**: `other` - 另一个操作数数组。  
  **返回**: 比较结果数组，1.0表示小于，0.0表示不小于。  
  **异常**: 当两个数组形状不一致时抛出 `IllegalArgumentException`。

- `boolean isLar(NdArray other)`  
  矩阵全元素大于比较，判断当前数组是否所有元素都大于另一个数组对应元素。  
  **参数**: `other` - 另一个操作数数组。  
  **返回**: 比较结果，true表示所有元素都大于，false表示存在不大于的元素。  
  **异常**: 当两个数组形状不一致时抛出 `IllegalArgumentException`。

### 基本数学函数
- `NdArray pow(Number number)`  
  幂运算，对数组每个元素进行幂运算。  
  **参数**: `number` - 幂指数。  
  **返回**: 幂运算结果数组。

- `NdArray square()`  
  平方运算，对数组每个元素进行平方运算。  
  **返回**: 平方运算结果数组。

- `NdArray sqrt()`  
  平方根运算，对数组每个元素进行开方运算。  
  **返回**: 平方根运算结果数组。

- `NdArray exp()`  
  自然指数运算，对数组每个元素进行e为底的指数运算。  
  **返回**: 指数运算结果数组。

- `NdArray sin()`  
  正弦函数运算，对数组每个元素进行sin运算。  
  **返回**: 正弦运算结果数组。

- `NdArray cos()`  
  余弦函数运算，对数组每个元素进行cos运算。  
  **返回**: 余弦运算结果数组。

- `NdArray tanh()`  
  双曲正切函数运算，对数组每个元素进行tanh运算。  
  **返回**: 双曲正切运算结果数组。

- `NdArray sigmoid()`  
  Sigmoid函数运算，对数组每个元素进行sigmoid运算。  
  **公式**: f(x) = 1 / (1 + e^(-x))。  
  **返回**: Sigmoid运算结果数组。

- `NdArray log()`  
  自然对数运算，对数组每个元素进行ln运算。  
  **返回**: 对数运算结果数组。  
  **异常**: 当输入值小于等于0时抛出 `ArithmeticException`。

- `NdArray softMax()`  
  Softmax函数运算，按行计算概率分布。  
  **公式**: softmax(x_i) = exp(x_i) / Σ(exp(x_j))，使用数值稳定版本实现，避免指数运算溢出。  
  **返回**: Softmax运算结果数组。  
  **异常**: 当数组不是矩阵时抛出 `IllegalArgumentException`。

- `NdArray maximum(Number number)`  
  元素级最大值运算，将数组中小于指定值的元素替换为该值。  
  **参数**: `number` - 阈值。  
  **返回**: 最大值运算结果数组。

- `NdArray mask(Number number)`  
  掩码运算，将数组中大于指定值的元素设为1，小于等于指定值的元素设为0。  
  **参数**: `number` - 阈值。  
  **返回**: 掩码运算结果数组。

### 张量变形操作
- `NdArray transpose()`  
  矩阵转置操作（二维矩阵），行列互换。  
  **返回**: 转置后的矩阵。  
  **异常**: 当数组不是矩阵时抛出 `IllegalArgumentException`。

- `NdArray transpose(int... order)`  
  多维数组转置操作，按指定维度顺序重新排列。  
  **参数**: `order` - 新的维度顺序。  
  **返回**: 转置后的数组。  
  **异常**: 当维度顺序无效时抛出 `IllegalArgumentException`。

- `NdArray reshape(Shape newShape)`  
  数组变形操作，改变数组形状但保持元素总数不变。  
  **参数**: `newShape` - 新的数组形状。  
  **返回**: 变形后的数组。  
  **异常**: 当新形状大小与原形状不匹配时抛出 `IllegalArgumentException`。

- `NdArray flatten()`  
  数组展平操作，将多维数组转换为一维行向量。  
  **返回**: 展平后的一维行向量。

### 统计和聚合操作
- `NdArray sum()`  
  元素累和运算，计算数组所有元素的总和。  
  **返回**: 所有元素的总和（标量）。

- `NdArray mean(int axis)`  
  矩阵均值运算，沿指定轴计算均值。  
  **参数**: `axis` - 聚合轴，axis=0表示按列计算均值，axis=1表示按行计算均值。  
  **返回**: 均值运算结果数组。

- `NdArray var(int axis)`  
  矩阵方差运算，沿指定轴计算方差。  
  **参数**: `axis` - 聚合轴，axis=0表示按列计算方差，axis=1表示按行计算方差。  
  **返回**: 方差运算结果数组。

- `NdArray sum(int axis)`  
  矩阵累和运算，沿指定轴计算累和。  
  **参数**: `axis` - 聚合轴，axis=0表示按列累和，axis=1表示按行累和。  
  **返回**: 累和运算结果数组。

- `NdArray sumTo(Shape _shape)`  
  按指定形状进行压缩累加运算，将当前数组按指定形状进行压缩，超出目标形状的部分会累加到对应位置。  
  **参数**: `_shape` - 目标形状。  
  **返回**: 压缩累加结果数组。  
  **异常**: 当数组不是矩阵或形状不合法时抛出 `IllegalArgumentException`。

- `NdArray broadcastTo(Shape _shape)`  
  数组广播运算，将当前数组广播到指定形状。  
  **参数**: `_shape` - 目标广播形状。  
  **返回**: 广播结果数组。  
  **异常**: 当数组不是矩阵或形状不合法时抛出 `IllegalArgumentException`。

- `NdArray argMax(int axis)`  
  沿指定轴查找最大值的索引。  
  **参数**: `axis` - 查找轴，axis=0表示按行查找每列的最大值索引，axis=1表示按列查找每行的最大值索引。  
  **返回**: 最大值索引数组。  
  **异常**: 当数组不是矩阵或轴参数无效时抛出 `IllegalArgumentException`。

- `NdArray dot(NdArray other)`  
  矩阵内积运算（矩阵乘法），执行标准的矩阵乘法运算，要求第一个矩阵的列数等于第二个矩阵的行数。  
  **参数**: `other` - 另一个矩阵。  
  **返回**: 矩阵乘法结果。  
  **异常**: 当数组不是矩阵或维度不匹配时抛出 `IllegalArgumentException`。

- `NdArray getItem(int[] _rowSlices, int[] _colSlices)`  
  获取数组的子集（切片操作）。  
  **参数**: `_rowSlices` - 行索引数组，null表示选择所有行；`_colSlices` - 列索引数组，null表示选择所有列。  
  **返回**: 切片结果数组。  
  **异常**: 当数组不是矩阵或参数不合法时抛出 `IllegalArgumentException`。

- `NdArray setItem(int[] _rowSlices, int[] _colSlices, float[] data)`  
  设置数组的子集（切片赋值操作）。  
  **参数**: `_rowSlices` - 行索引数组，null表示选择所有行；`_colSlices` - 列索引数组，null表示选择所有列；`data` - 要设置的数据。  
  **返回**: 当前数组实例。  
  **异常**: 当数组不是矩阵或参数不合法时抛出 `IllegalArgumentException`。

- `NdArray max(int axis)`  
  沿指定轴查找最大值。  
  **参数**: `axis` - 查找轴，axis=0表示按行查找每列的最大值，axis=1表示按列查找每行的最大值。  
  **返回**: 最大值数组。  
  **异常**: 当数组不是矩阵或轴参数无效时抛出 `IllegalArgumentException`。

- `NdArray min(int axis)`  
  沿指定轴查找最小值。  
  **参数**: `axis` - 查找轴，axis=0表示按行查找每列的最小值，axis=1表示按列查找每行的最小值。  
  **返回**: 最小值数组。  
  **异常**: 当数组不是矩阵或轴参数无效时抛出 `IllegalArgumentException`。

- `float max()`  
  查找数组中的最大值（全局最大值）。  
  **返回**: 数组中的最大值。

- `NdArray subNdArray(int startRow, int endRow, int startCol, int endCol)`  
  获取子数组（矩阵的子区域）。  
  **参数**: `startRow` - 起始行索引（包含）；`endRow` - 结束行索引（不包含）；`startCol` - 起始列索引（包含）；`endCol` - 结束列索引（不包含）。  
  **返回**: 子数组。  
  **异常**: 当数组不是矩阵时抛出 `IllegalArgumentException`。

- `NdArray addAt(int[] rowSlices, int[] colSlices, NdArray other)`  
  在指定位置累加数组元素，在指定的行和列位置上累加另一个数组的元素。  
  **参数**: `rowSlices` - 行索引数组，指定要累加的行位置；`colSlices` - 列索引数组，指定要累加的列位置；`other` - 要累加的数组。  
  **返回**: 累加结果数组。  
  **异常**: 当输入参数不合法时抛出 `IllegalArgumentException`；当数组不是矩阵时抛出 `RuntimeException`。

- `NdArray addTo(int i, int j, NdArray other)`  
  将另一个数组累加到当前数组的指定位置。  
  **参数**: `i` - 起始行索引；`j` - 起始列索引；`other` - 要累加的数组。  
  **返回**: 当前数组实例。  
  **异常**: 当数组不是矩阵时抛出 `IllegalArgumentException`。

- `NdArray clip(float min, float max)`  
  裁剪数组元素到指定范围，将数组中小于最小值的元素设为最小值，大于最大值的元素设为最大值。  
  **参数**: `min` - 最小值；`max` - 最大值。  
  **返回**: 裁剪后的数组。  
  **异常**: 当最小值大于最大值时抛出 `IllegalArgumentException`。

### 辅助方法
- `void fillAll(Number number)`  
  用指定值填充整个数组。  
  **参数**: `number` - 填充值。

- `Number getNumber()`  
  获取数组的第一个元素值（标量值）。  
  **返回**: 第一个元素值。

- `Shape getShape()`  
  获取数组的形状。  
  **返回**: 数组形状。

- `void setShape(Shape shape)`  
  设置数组的形状。  
  **参数**: `shape` - 新形状。  
  **异常**: 当新形状大小与当前形状不匹配时抛出 `IllegalArgumentException`。

- `float[][] getMatrix()`  
  将数组转换为二维数组（矩阵）返回。  
  **返回**: 二维数组表示。  
  **异常**: 当数组维度大于2时抛出 `IllegalArgumentException`。

- `float[][][] get3dArray()`  
  将数组转换为三维数组返回。  
  **返回**: 三维数组表示。  
  **异常**: 当数组不是三维时抛出 `IllegalArgumentException`。

- `float[][][][] get4dArray()`  
  将数组转换为四维数组返回。  
  **返回**: 四维数组表示。  
  **异常**: 当数组不是四维时抛出 `IllegalArgumentException`。

- `String toString()`  
  优化的toString方法，提供数组的字符串表示。对于小数组会显示所有元素，对于大数组只会显示部分元素。  
  **返回**: 数组的字符串表示。

- `boolean equals(Object obj)`  
  优化的equals方法，比较两个NdArray对象是否相等。  
  **参数**: `obj` - 另一个对象。  
  **返回**: 是否相等。

- `int hashCode()`  
  优化的hashCode方法，为NdArray对象生成哈希码。  
  **返回**: 哈希码。

- `void set(float value, int... _dimension)`  
  按维度下标设置某一个值。  
  **参数**: `value` - 要设置的值；`_dimension` - 维度下标数组。  
  **异常**: 当维度数量不匹配时抛出 `IllegalArgumentException`。

- `float get(int... _dimension)`  
  按维度下标获取某一个值。  
  **参数**: `_dimension` - 维度下标数组。  
  **返回**: 对应位置的值。  
  **异常**: 当维度数量不匹配时抛出 `IllegalArgumentException`。

**Section sources**
- [NdArray.java](file://src/main/java/io/leavesfly/tinydl/ndarr/NdArray.java#L1-L1802)

## MoE相关API
本节介绍框架中新增的MoE（Mixture of Experts）相关API，包括 `MoEGPTModel` 和 `MoETransformerBlock` 类。

### MoEGPTModel 类
`MoEGPTModel` 类是一个基于Transformer架构的MoE-GPT模型，通过引入专家混合机制来提高模型的容量和效率。

#### 构造方法
- `MoEGPTModel(String name, int vocabSize, int dModel, int numLayers, int numHeads, int numExperts, int topK, int expertHiddenDim, int maxSeqLength, double dropoutRate, double loadBalancingWeight)`  
  构造MoE-GPT模型。  
  **参数**: `name` - 模型名称；`vocabSize` - 词汇表大小；`dModel` - 模型维度；`numLayers` - Transformer层数；`numHeads` - 注意力头数；`numExperts` - 每层的专家数量；`topK` - Top-K专家选择数量；`expertHiddenDim` - 专家隐藏维度；`maxSeqLength` - 最大序列长度；`dropoutRate` - Dropout比率；`loadBalancingWeight` - 负载均衡权重。  
  **异常**: 当dModel不能被numHeads整除或topK大于numExperts时抛出 `IllegalArgumentException`。

#### 工厂方法
- `static MoEGPTModel createMediumModel(String name, int vocabSize)`  
  创建中等规模MoE-GPT模型的工厂方法。  
  **参数**: `name` - 模型名称；`vocabSize` - 词汇表大小。  
  **返回**: MoE-GPT模型实例。

- `static MoEGPTModel createSmallModel(String name, int vocabSize)`  
  创建小规模MoE-GPT模型的工厂方法，适合实验和快速原型。  
  **参数**: `name` - 模型名称；`vocabSize` - 词汇表大小。  
  **返回**: MoE-GPT模型实例。

- `static MoEGPTModel createTinyModel(String name, int vocabSize)`  
  创建微型MoE-GPT模型的工厂方法，用于调试和概念验证。  
  **参数**: `name` - 模型名称；`vocabSize` - 词汇表大小。  
  **返回**: MoE-GPT模型实例。

#### 核心方法
- `void init()`  
  初始化模型，包括Token嵌入层、MoE Transformer块、最终层归一化和输出头。  
  **说明**: 打印初始化信息和参数统计。

- `Variable layerForward(Variable... inputs)`  
  执行模型的前向传播。  
  **参数**: `inputs` - 输入变量数组。  
  **返回**: 输出变量。

- `Variable generate(Variable tokenIds)`  
  生成文本的前向传播（用于推理）。  
  **参数**: `tokenIds` - 输入token序列。  
  **返回**: 下一个token的概率分布。

- `int predictNextToken(NdArray tokenIds)`  
  预测下一个token（贪心解码）。  
  **参数**: `tokenIds` - 输入token序列。  
  **返回**: 最可能的下一个token ID。

- `float computeTotalLoadBalancingLoss()`  
  计算总的负载均衡损失，这个损失应该添加到训练损失中以鼓励专家的均匀使用。  
  **返回**: 总负载均衡损失。

- `void resetAllExpertStatistics()`  
  重置所有MoE块的专家使用统计。

- `void printAllExpertStatistics()`  
  打印所有层的专家使用统计。

- `List<float[]> getAllLayersExpertUsageRates()`  
  获取各层专家使用率的汇总统计。  
  **返回**: 每层的专家使用率数组。

- `long getParameterCount()`  
  计算模型的总参数量。  
  **返回**: 模型总参数量。

- `long getActiveParameterCount()`  
  计算有效参数量（考虑MoE的稀疏性）。  
  **返回**: 每次前向传播实际使用的参数量。

- `void printModelInfo()`  
  打印模型信息，包括参数统计和效率。

**Section sources**
- [MoEGPTModel.java](file://src/main/java/io/leavesfly/tinydl/modality/nlp/MoEGPTModel.java#L35-L512)

### MoETransformerBlock 类
`MoETransformerBlock` 类是一个支持MoE的Transformer Block，通过引入MoE层来替代传统的FeedForward层。

#### 构造方法
- `MoETransformerBlock(String name, int dModel, int numHeads, int numExperts, int topK, int expertHiddenDim, double dropoutRate, double loadBalancingWeight)`  
  构造支持MoE的Transformer Block。  
  **参数**: `name` - 块名称；`dModel` - 模型维度；`numHeads` - 注意力头数；`numExperts` - 专家数量；`topK` - Top-K专家选择数量；`expertHiddenDim` - 专家隐藏层维度；`dropoutRate` - Dropout比率；`loadBalancingWeight` - 负载均衡权重。

- `MoETransformerBlock(String name, int dModel, int numHeads, int numExperts)`  
  简化构造函数，使用默认参数。  
  **参数**: `name` - 块名称；`dModel` - 模型维度；`numHeads` - 注意力头数；`numExperts` - 专家数量。

- `MoETransformerBlock(String name, int dModel, int numHeads, int numExperts, int topK)`  
  带Top-K参数的构造函数。  
  **参数**: `name` - 块名称；`dModel` - 模型维度；`numHeads` - 注意力头数；`numExperts` - 专家数量；`topK` - Top-K专家选择数量。

#### 核心方法
- `void init()`  
  初始化块，包括第一个层归一化、多头注意力、第二个层归一化和MoE层。

- `Variable layerForward(Variable... inputs)`  
  执行块的前向传播。  
  **参数**: `inputs` - 输入变量数组。  
  **返回**: 输出变量。

- `float getLoadBalancingLoss()`  
  获取负载均衡损失，这个损失可以添加到总的训练损失中，以鼓励专家的均匀使用。  
  **返回**: 负载均衡损失值。

- `void resetExpertStatistics()`  
  重置MoE专家使用统计。

- `void printExpertStatistics()`  
  打印专家使用统计信息。

- `float[] getExpertUsageRates()`  
  获取专家使用率。  
  **返回**: 专家使用率数组。

- `long getParameterCount()`  
  计算Block的总参数量。  
  **返回**: 总参数量。

**Section sources**
- [MoETransformerBlock.java](file://src/main/java/io/leavesfly/tinydl/modality/nlp/block/MoETransformerBlock.java#L32-L390)

## 强化学习相关API
本节介绍框架中新增的强化学习相关API，包括 `Agent` 和 `BanditAgent` 抽象类。

### Agent 抽象类
`Agent` 类是所有智能体的基类，定义了智能体的基本行为和属性。

#### 字段
- `protected String name`  
  智能体名称。

- `protected int stateDim`  
  状态空间维度。

- `protected int actionDim`  
  动作空间维度。

- `protected Model model`  
  主要的神经网络模型。

- `protected float learningRate`  
  学习率。

- `protected float epsilon`  
  探索率（epsilon-greedy策略中的epsilon）。

- `protected float gamma`  
  折扣因子。

- `protected int trainingStep`  
  训练步数计数器。

- `protected boolean training`  
  是否处于训练模式。

#### 构造方法
- `Agent(String name, int stateDim, int actionDim, float learningRate, float epsilon, float gamma)`  
  构造函数。  
  **参数**: `name` - 智能体名称；`stateDim` - 状态空间维度；`actionDim` - 动作空间维度；`learningRate` - 学习率；`epsilon` - 初始探索率；`gamma` - 折扣因子。

#### 核心方法
- `public abstract Variable selectAction(Variable state)`  
  根据当前状态选择动作。  
  **参数**: `state` - 当前状态。  
  **返回**: 选择的动作。

- `public abstract void learn(Experience experience)`  
  从经验中学习更新模型。  
  **参数**: `experience` - 经验数据。

- `public abstract void learnBatch(Experience[] experiences)`  
  批量学习更新模型。  
  **参数**: `experiences` - 经验批次。

- `public abstract void storeExperience(Experience experience)`  
  存储经验（用于经验回放）。  
  **参数**: `experience` - 要存储的经验。

- `public Map<String, Parameter> getAllParams()`  
  获取模型的所有参数。  
  **返回**: 参数映射。

- `public void clearGrads()`  
  清空梯度。

- `public void setTraining(boolean training)`  
  设置训练模式。  
  **参数**: `training` - 是否为训练模式。

- `public float getEpsilon()`  
  获取当前探索率。  
  **返回**: 当前探索率。

- `public void setEpsilon(float epsilon)`  
  设置探索率。  
  **参数**: `epsilon` - 新的探索率。

- `public void decayEpsilon(float decayRate)`  
  衰减探索率。  
  **参数**: `decayRate` - 衰减率。

- `public int getTrainingStep()`  
  获取训练步数。  
  **返回**: 训练步数。

- `protected void incrementTrainingStep()`  
  增加训练步数。

- `public String getName()`  
  获取智能体名称。  
  **返回**: 智能体名称。

- `public int getStateDim()`  
  获取状态空间维度。  
  **返回**: 状态空间维度。

- `public int getActionDim()`  
  获取动作空间维度。  
  **返回**: 动作空间维度。

- `public float getLearningRate()`  
  获取学习率。  
  **返回**: 学习率。

- `public void setLearningRate(float learningRate)`  
  设置学习率。  
  **参数**: `learningRate` - 新的学习率。

- `public float getGamma()`  
  获取折扣因子。  
  **返回**: 折扣因子。

- `public void reset()`  
  重置智能体状态。

- `public abstract void saveModel(String filepath)`  
  保存模型参数。  
  **参数**: `filepath` - 保存路径。

- `public abstract void loadModel(String filepath)`  
  加载模型参数。  
  **参数**: `filepath` - 加载路径。

**Section sources**
- [Agent.java](file://src/main/java/io/leavesfly/tinydl/modality/rl/Agent.java#L19-L265)

### BanditAgent 抽象类
`BanditAgent` 类是多臂老虎机智能体的基类，继承自 `Agent`。

#### 字段
- `protected int[] actionCounts`  
  每个臂被选择的次数。

- `protected float[] totalRewards`  
  每个臂的累积奖励。

- `protected float[] estimatedRewards`  
  每个臂的估计平均奖励（Q值）。

- `protected int totalActions`  
  总的动作选择次数。

#### 构造方法
- `BanditAgent(String name, int numArms)`  
  构造函数。  
  **参数**: `name` - 智能体名称；`numArms` - 臂的数量。

#### 核心方法
- `public abstract Variable selectAction(Variable state)`  
  根据当前状态选择动作（多臂老虎机不依赖状态）。  
  **参数**: `state` - 当前状态（在多臂老虎机中会被忽略）。  
  **返回**: 选择的动作（臂的索引）。

- `public abstract int selectArm()`  
  多臂老虎机专用的动作选择方法。  
  **返回**: 选择的臂的索引。

- `public void learn(Experience experience)`  
  从经验中学习更新模型。  
  **参数**: `experience` - 经验数据（包含选择的动作和获得的奖励）。

- `protected void updateStatistics(int armIndex, float reward)`  
  更新统计信息。  
  **参数**: `armIndex` - 选择的臂索引；`reward` - 获得的奖励。

- `public float getEstimatedReward(int armIndex)`  
  获取指定臂的估计奖励。  
  **参数**: `armIndex` - 臂索引。  
  **返回**: 估计奖励。

- `public int getActionCount(int armIndex)`  
  获取指定臂被选择的次数。  
  **参数**: `armIndex` - 臂索引。  
  **返回**: 选择次数。

- `public float getTotalReward(int armIndex)`  
  获取指定臂的累积奖励。  
  **参数**: `armIndex` - 臂索引。  
  **返回**: 累积奖励。

- `public float[] getAllEstimatedRewards()`  
  获取所有臂的估计奖励。  
  **返回**: 估计奖励数组。

- `public int[] getAllActionCounts()`  
  获取所有臂的选择次数。  
  **返回**: 选择次数数组。

- `public int getTotalActions()`  
  获取总的动作选择次数。  
  **返回**: 总选择次数。

- `public int getBestArmIndex()`  
  获取当前最优臂的索引（基于估计奖励）。  
  **返回**: 最优臂索引。

- `public float getBestEstimatedReward()`  
  获取当前最优臂的估计奖励。  
  **返回**: 最优臂估计奖励。

- `public float getOptimalActionRate()`  
  计算选择概率最高的臂被选择的频率。  
  **返回**: 最优选择频率。

- `public void reset()`  
  重置智能体的统计信息。

- `public void learnBatch(Experience[] experiences)`  
  批量学习更新模型（多臂老虎机通常不需要批量学习）。  
  **参数**: `experiences` - 经验批次。

- `public void storeExperience(Experience experience)`  
  存储经验（多臂老虎机通常不需要经验回放）。  
  **参数**: `experience` - 要存储的经验。

- `public void saveModel(String filepath)`  
  保存模型参数（多臂老虎机保存统计信息）。  
  **参数**: `filepath` - 保存路径。

- `public void loadModel(String filepath)`  
  加载模型参数（多臂老虎机加载统计信息）。  
  **参数**: `filepath` - 加载路径。

- `public void printStatus()`  
  打印智能体当前状态。

**Section sources**
- [BanditAgent.java](file://src/main/java/io/leavesfly/tinydl/modality/rl/agent/BanditAgent.java#L17-L273)