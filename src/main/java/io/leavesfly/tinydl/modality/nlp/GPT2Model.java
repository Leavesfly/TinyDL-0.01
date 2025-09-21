package io.leavesfly.tinydl.modality.nlp;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Block;
import io.leavesfly.tinydl.nnet.block.transformer.GPT2Block;
import io.leavesfly.tinydl.nnet.layer.transformer.GPT2OutputHead;
import io.leavesfly.tinydl.nnet.layer.transformer.GPT2TokenEmbedding;
import io.leavesfly.tinydl.nnet.layer.transformer.LayerNorm;

import java.util.ArrayList;
import java.util.List;

/**
 * GPT-2 小规模语言模型实现
 * 
 * GPT-2是一个基于Transformer解码器的自回归语言模型，特点：
 * 1. 仅使用解码器架构
 * 2. 使用掩码多头自注意力防止未来信息泄露
 * 3. Pre-LayerNorm结构
 * 4. 残差连接
 * 
 * 模型结构：
 * Token Embedding + Position Embedding
 * → N × GPT2Block
 * → Final LayerNorm
 * → Output Head
 */
public class GPT2Model extends Block {
    
    // 模型组件
    private GPT2TokenEmbedding tokenEmbedding;    // Token嵌入层
    private List<GPT2Block> transformerBlocks;    // Transformer块列表
    private LayerNorm finalLayerNorm;             // 最终层归一化
    private GPT2OutputHead outputHead;            // 输出头
    
    // 模型参数
    private int vocabSize;      // 词汇表大小
    private int dModel;         // 模型维度
    private int numLayers;      // Transformer层数
    private int numHeads;       // 注意力头数
    private int dFF;            // 前馈网络隐藏维度
    private int maxSeqLength;   // 最大序列长度
    private double dropoutRate; // Dropout比率
    
    /**
     * 构造GPT-2模型
     * 
     * @param name 模型名称
     * @param vocabSize 词汇表大小
     * @param dModel 模型维度
     * @param numLayers Transformer层数
     * @param numHeads 注意力头数
     * @param dFF 前馈网络隐藏维度
     * @param maxSeqLength 最大序列长度
     * @param dropoutRate Dropout比率
     */
    public GPT2Model(String name, int vocabSize, int dModel, int numLayers, 
                     int numHeads, int dFF, int maxSeqLength, double dropoutRate) {
        super(name, new Shape(-1, maxSeqLength), new Shape(-1, maxSeqLength, vocabSize));
        
        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException("dModel must be divisible by numHeads");
        }
        
        this.vocabSize = vocabSize;
        this.dModel = dModel;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.dFF = dFF;
        this.maxSeqLength = maxSeqLength;
        this.dropoutRate = dropoutRate;
        
        init();
    }
    
    /**
     * 创建小规模GPT-2模型的构造函数
     * 默认参数适用于实验和学习
     */
    public static GPT2Model createSmallModel(String name, int vocabSize) {
        return new GPT2Model(
            name,
            vocabSize,    // 词汇表大小
            256,          // 嵌入维度
            6,            // 6层Transformer
            8,            // 8个注意力头
            1024,         // 前馈网络隐藏维度
            512,          // 最大序列长度
            0.1           // Dropout比率
        );
    }
    
    /**
     * 创建微型GPT-2模型的构造函数
     * 用于快速实验和调试
     */
    public static GPT2Model createTinyModel(String name, int vocabSize) {
        return new GPT2Model(
            name,
            vocabSize,    // 词汇表大小
            128,          // 嵌入维度
            4,            // 4层Transformer
            4,            // 4个注意力头
            512,          // 前馈网络隐藏维度
            128,          // 最大序列长度
            0.1           // Dropout比率
        );
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            System.out.println("Initializing GPT-2 Model with:");
            System.out.println("  Vocab Size: " + vocabSize);
            System.out.println("  Model Dim: " + dModel);
            System.out.println("  Layers: " + numLayers);
            System.out.println("  Heads: " + numHeads);
            System.out.println("  Max Seq Length: " + maxSeqLength);
            
            // 1. 初始化Token嵌入层
            tokenEmbedding = new GPT2TokenEmbedding(
                name + "_token_embedding", 
                vocabSize, 
                dModel, 
                maxSeqLength, 
                true,          // 使用学习的位置嵌入
                dropoutRate
            );
            addLayer(tokenEmbedding);
            
            // 2. 初始化Transformer块
            transformerBlocks = new ArrayList<>();
            for (int i = 0; i < numLayers; i++) {
                GPT2Block block = new GPT2Block(
                    name + "_block_" + i,
                    dModel,
                    numHeads,
                    dFF,
                    dropoutRate
                );
                transformerBlocks.add(block);
                addLayer(block);
            }
            
            // 3. 初始化最终层归一化
            finalLayerNorm = new LayerNorm(name + "_final_ln", dModel);
            addLayer(finalLayerNorm);
            
            // 4. 初始化输出头
            outputHead = new GPT2OutputHead(name + "_output_head", dModel, vocabSize);
            addLayer(outputHead);
            
            alreadyInit = true;
            System.out.println("GPT-2 Model initialization completed.");
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable tokenIds = inputs[0];  // shape: (batch_size, seq_len)
        
        // 验证输入形状
        NdArray inputData = tokenIds.getValue();
        if (inputData.shape.dimension.length != 2) {
            throw new IllegalArgumentException("Input must be 2D tensor (batch_size, seq_len)");
        }
        
        int seqLen = inputData.shape.dimension[1];
        if (seqLen > maxSeqLength) {
            throw new IllegalArgumentException(
                String.format("Input sequence length %d exceeds maximum %d", seqLen, maxSeqLength)
            );
        }
        
        // 1. Token嵌入 + 位置嵌入
        Variable x = tokenEmbedding.layerForward(tokenIds);
        
        // 2. 通过所有Transformer块
        for (GPT2Block block : transformerBlocks) {
            x = block.layerForward(x);
        }
        
        // 3. 最终层归一化
        x = finalLayerNorm.layerForward(x);
        
        // 4. 输出投影到词汇表
        Variable logits = outputHead.layerForward(x);
        
        return logits;
    }
    
    /**
     * 生成文本的前向传播（用于推理）
     * @param tokenIds 输入token序列
     * @return 下一个token的概率分布
     */
    public Variable generate(Variable tokenIds) {
        return layerForward(tokenIds);
    }
    
    /**
     * 预测下一个token
     * @param tokenIds 输入token序列
     * @return 最可能的下一个token ID
     */
    public int predictNextToken(NdArray tokenIds) {
        Variable input = new Variable(tokenIds);
        Variable logits = generate(input);
        
        // 获取最后一个位置的logits
        NdArray logitsData = logits.getValue();
        int batchSize = logitsData.shape.dimension[0];
        int seqLen = logitsData.shape.dimension[1];
        
        // 找到最大概率的token（简单的贪心解码）
        float maxLogit = Float.NEGATIVE_INFINITY;
        int bestToken = 0;
        
        for (int v = 0; v < vocabSize; v++) {
            float logit = logitsData.get(0, seqLen - 1, v);  // 取第一个batch的最后一个位置
            if (logit > maxLogit) {
                maxLogit = logit;
                bestToken = v;
            }
        }
        
        return bestToken;
    }
    
    /**
     * 计算模型的总参数量
     */
    public long getParameterCount() {
        long totalParams = 0;
        
        // Token嵌入参数
        totalParams += (long) vocabSize * dModel;
        
        // 位置嵌入参数
        totalParams += (long) maxSeqLength * dModel;
        
        // 每个Transformer块的参数
        long blockParams = 0;
        // 注意力权重: 3 * (dModel * dModel) + dModel * dModel
        blockParams += 4L * dModel * dModel;
        // 前馈网络: dModel * dFF + dFF * dModel
        blockParams += 2L * dModel * dFF;
        // 层归一化参数: 2 * (2 * dModel)
        blockParams += 4L * dModel;
        
        totalParams += blockParams * numLayers;
        
        // 最终层归一化参数
        totalParams += 2L * dModel;
        
        // 输出头参数
        totalParams += (long) dModel * vocabSize;
        
        return totalParams;
    }
    
    /**
     * 打印模型信息
     */
    public void printModelInfo() {
        System.out.println("=== GPT-2 Model Information ===");
        System.out.println("Model Name: " + name);
        System.out.println("Vocabulary Size: " + vocabSize);
        System.out.println("Model Dimension: " + dModel);
        System.out.println("Number of Layers: " + numLayers);
        System.out.println("Number of Heads: " + numHeads);
        System.out.println("Feed Forward Dimension: " + dFF);
        System.out.println("Max Sequence Length: " + maxSeqLength);
        System.out.println("Dropout Rate: " + dropoutRate);
        System.out.println("Total Parameters: " + getParameterCount());
        System.out.println("==============================");
    }
    
    // Getters
    public GPT2TokenEmbedding getTokenEmbedding() { return tokenEmbedding; }
    public List<GPT2Block> getTransformerBlocks() { return transformerBlocks; }
    public LayerNorm getFinalLayerNorm() { return finalLayerNorm; }
    public GPT2OutputHead getOutputHead() { return outputHead; }
    public int getVocabSize() { return vocabSize; }
    public int getDModel() { return dModel; }
    public int getNumLayers() { return numLayers; }
    public int getNumHeads() { return numHeads; }
    public int getDFF() { return dFF; }
    public int getMaxSeqLength() { return maxSeqLength; }
    public double getDropoutRate() { return dropoutRate; }
}