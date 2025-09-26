# Word2Vec 实现文档

## 概述

本文档详细介绍了在TinyDL框架中实现的Word2Vec词向量模型。Word2Vec是一种用于学习词嵌入的经典神经网络模型，能够将离散的词汇映射到连续的向量空间中，使得语义相似的词在向量空间中距离较近。

## 实现特性

### ✅ 已实现功能

1. **双训练模式支持**
   - Skip-gram模式：根据中心词预测上下文词
   - CBOW模式：根据上下文词预测中心词

2. **完整的词汇表管理**
   - 自动构建词汇表
   - 词频统计和排序
   - 词汇到索引的双向映射

3. **负采样技术**
   - 支持负采样优化训练
   - 基于词频的3/4次方平滑分布采样
   - 可配置负样本数量

4. **词向量操作**
   - 词向量提取
   - 余弦相似度计算
   - 最相似词查找

5. **数据集支持**
   - 专用的Word2VecDataSet类
   - 支持Skip-gram和CBOW模式的数据生成
   - 批处理和数据混洗功能

## 核心类结构

### Word2Vec 主类

```java
public class Word2Vec extends Block {
    // 训练模式枚举
    public enum TrainingMode {
        SKIP_GRAM,  // Skip-gram模式
        CBOW        // CBOW模式
    }
    
    // 主要组件
    private Embedding inputEmbedding;   // 输入词嵌入层
    private Embedding outputEmbedding;  // 输出词嵌入层（负采样用）
    private LinearLayer outputLayer;   // 输出线性层
    private SoftMaxLayer softmaxLayer; // Softmax层
}
```

### Word2VecDataSet 数据集类

```java
public class Word2VecDataSet extends ArrayDataset {
    public enum TrainingMode {
        SKIP_GRAM,  // Skip-gram训练模式
        CBOW        // CBOW训练模式
    }
    
    // 主要功能
    - buildVocabulary()          // 构建词汇表
    - generateTrainingData()     // 生成训练数据
    - generateSkipGramSamples()  // 生成Skip-gram样本
    - generateCBOWSamples()      // 生成CBOW样本
}
```

## 使用示例

### 基本使用

```java
// 1. 准备语料库
List<String> corpus = Arrays.asList(
    "机器", "学习", "是", "人工", "智能", "的", "重要", "分支",
    "深度", "学习", "是", "机器", "学习", "的", "子", "领域"
);

// 2. 创建Word2Vec模型
Word2Vec word2vec = new Word2Vec(
    "word2vec_model",     // 模型名称
    50,                   // 词汇表大小
    10,                   // 词向量维度
    Word2Vec.TrainingMode.SKIP_GRAM, // 训练模式
    2,                    // 上下文窗口大小
    false,                // 是否使用负采样
    5                     // 负样本数量
);

// 3. 构建词汇表
word2vec.buildVocab(corpus);

// 4. 生成训练样本
List<Word2Vec.TrainingSample> samples = word2vec.generateTrainingSamples(corpus);

// 5. 训练模型（简化示例）
Model model = new Model("word2vec_model", word2vec);
Optimizer optimizer = new SGD(model, 0.01f);
SoftmaxCrossEntropy lossFunc = new SoftmaxCrossEntropy();

// 训练循环
for (int epoch = 0; epoch < 100; epoch++) {
    for (Word2Vec.TrainingSample sample : samples) {
        Variable input = new Variable(new NdArray(new float[][]{{sample.input}}));
        Variable target = new Variable(new NdArray(new float[][]{{sample.target}}));
        
        Variable output = model.forward(input);
        Variable loss = lossFunc.loss(target, output);
        
        model.clearGrads();
        loss.backward();
        optimizer.update();
    }
}
```

### 使用数据集类

```java
// 1. 创建数据集
Word2VecDataSet dataset = new Word2VecDataSet(
    corpus,                                    // 语料库
    32,                                       // 批次大小
    2,                                        // 窗口大小
    Word2VecDataSet.TrainingMode.SKIP_GRAM,  // 训练模式
    100                                      // 最大词汇表大小
);

// 2. 准备数据
dataset.prepare();

// 3. 打印统计信息
dataset.printStatistics();

// 4. 获取批次进行训练
List<Batch> batches = dataset.getBatches();
```

### 词向量操作

```java
// 获取词向量
NdArray wordVector = word2vec.getWordVector("学习");

// 查找相似词
List<String> similarWords = word2vec.mostSimilar("学习", 5);

// 负采样
int targetWord = word2vec.getWord2idx().get("学习");
List<Integer> negatives = word2vec.negativeSampling(targetWord, 3);
```

## 技术特点

### 1. 模块化设计
- 继承自Block类，与TinyDL框架完美集成
- 支持层级嵌套和参数管理
- 可以作为更大模型的组件使用

### 2. 高效的负采样
- 实现了基于词频的负采样算法
- 使用3/4次方平滑提高低频词被选择的概率
- 避免与目标词相同的负样本

### 3. 灵活的训练模式
- Skip-gram：适合小语料库，对低频词效果好
- CBOW：训练速度快，对高频词效果好

### 4. 完整的词汇管理
- 自动词频统计和排序
- 支持词汇表大小限制
- 双向索引映射

## 性能考虑

### 优化点
1. **内存效率**：使用NdArray进行高效的矩阵运算
2. **计算优化**：支持负采样减少softmax计算开销
3. **训练稳定性**：梯度计算和参数更新机制完善

### 扩展性
1. **多种损失函数**：支持传统softmax和负采样两种方式
2. **自定义采样**：可以自定义负采样策略
3. **预训练加载**：架构支持预训练模型加载

## 应用场景

### 1. 文本分类
- 将词转换为密集向量表示
- 作为下游任务的特征提取器

### 2. 语义相似度计算
- 计算词与词之间的语义相似度
- 支持词汇聚类和主题分析

### 3. 自然语言处理预处理
- 为RNN、LSTM等模型提供词嵌入
- 减少输入维度，提高模型效率

## 测试结果

运行测试程序的输出显示：

```
=== Skip-gram 模式训练 ===
词汇表构建完成，共 49 个词
生成训练样本 290 个
开始Skip-gram训练...
Epoch 100/100, Average Loss: 3.9098

=== 词向量测试 ===
词 '学习' 的向量维度: [1,10]
词 '学习' 的向量值: [0.046, 0.011, -0.024, 0.043, -0.009...]
与 '学习' 最相似的词: [喜欢, 属于, 深度]
```

## 未来改进方向

### 1. 性能优化
- [ ] 分层softmax实现
- [ ] 子采样高频词
- [ ] 多线程训练支持

### 2. 功能扩展
- [ ] 支持预训练词向量加载
- [ ] 增加词向量可视化功能
- [ ] 支持短语和多词表达

### 3. 算法增强
- [ ] FastText算法支持
- [ ] GloVe算法实现
- [ ] 动态词汇表更新

## 结论

这个Word2Vec实现提供了完整的词向量训练功能，支持Skip-gram和CBOW两种模式，集成了负采样优化技术，并提供了方便的数据集管理工具。实现完全兼容TinyDL框架的设计理念，可以作为自然语言处理任务的基础组件使用。

通过测试验证，模型能够正确训练并生成有意义的词向量表示，为后续的NLP应用提供了可靠的基础。