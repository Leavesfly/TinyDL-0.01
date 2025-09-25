package io.leavesfly.tinydl.modality.nlp;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.modality.nlp.block.MoETransformerBlock;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Block;
import io.leavesfly.tinydl.nnet.layer.transformer.GPT2OutputHead;
import io.leavesfly.tinydl.nnet.layer.transformer.GPT2TokenEmbedding;
import io.leavesfly.tinydl.nnet.layer.transformer.LayerNorm;

import java.util.ArrayList;
import java.util.List;

/**
 * 基于Mixture of Experts (MoE) 的GPT模型实现
 * 
 * 这个模型将传统GPT-2中的FeedForward层替换为MoE层，
 * 从而大幅增加模型容量而不显著增加计算开销。
 * 
 * MoE-GPT模型的核心优势：
 * 1. 大幅增加模型参数量而保持合理的计算成本
 * 2. 每个token只激活部分专家，实现稀疏计算
 * 3. 不同专家可以专门处理不同类型的语言模式
 * 4. 可以通过增加专家数量来扩展模型容量
 * 
 * 模型结构：
 * Token Embedding + Position Embedding
 * → N × MoETransformerBlock  (替换标准TransformerBlock)
 * → Final LayerNorm
 * → Output Head
 * 
 * @author leavesfly
 * @version 0.01
 */
public class MoEGPTModel extends Block {
    
    /**
     * Token嵌入层
     */
    private GPT2TokenEmbedding tokenEmbedding;
    
    /**
     * MoE Transformer块列表
     */
    private List<MoETransformerBlock> moeTransformerBlocks;
    
    /**
     * 最终层归一化
     */
    private LayerNorm finalLayerNorm;
    
    /**
     * 输出头
     */
    private GPT2OutputHead outputHead;
    
    /**
     * 词汇表大小
     */
    private int vocabSize;
    
    /**
     * 模型维度
     */
    private int dModel;
    
    /**
     * Transformer层数
     */
    private int numLayers;
    
    /**
     * 注意力头数
     */
    private int numHeads;
    
    /**
     * 专家数量
     */
    private int numExperts;
    
    /**
     * Top-K专家选择
     */
    private int topK;
    
    /**
     * 专家隐藏维度
     */
    private int expertHiddenDim;
    
    /**
     * 最大序列长度
     */
    private int maxSeqLength;
    
    /**
     * Dropout比率
     */
    private double dropoutRate;
    
    /**
     * 负载均衡权重
     */
    private double loadBalancingWeight;
    
    /**
     * 总负载均衡损失（累积所有MoE层的损失）
     */
    private float totalLoadBalancingLoss;
    
    /**
     * 构造MoE-GPT模型
     * 
     * @param name 模型名称
     * @param vocabSize 词汇表大小
     * @param dModel 模型维度
     * @param numLayers Transformer层数
     * @param numHeads 注意力头数
     * @param numExperts 每层的专家数量
     * @param topK Top-K专家选择数量
     * @param expertHiddenDim 专家隐藏维度
     * @param maxSeqLength 最大序列长度
     * @param dropoutRate Dropout比率
     * @param loadBalancingWeight 负载均衡权重
     */
    public MoEGPTModel(String name, int vocabSize, int dModel, int numLayers, 
                       int numHeads, int numExperts, int topK, int expertHiddenDim,
                       int maxSeqLength, double dropoutRate, double loadBalancingWeight) {
        super(name, new Shape(-1, maxSeqLength), new Shape(-1, maxSeqLength, vocabSize));
        
        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException("dModel必须能被numHeads整除");
        }
        
        if (topK > numExperts) {
            throw new IllegalArgumentException("topK不能大于numExperts");
        }
        
        this.vocabSize = vocabSize;
        this.dModel = dModel;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.numExperts = numExperts;
        this.topK = topK;
        this.expertHiddenDim = expertHiddenDim;
        this.maxSeqLength = maxSeqLength;
        this.dropoutRate = dropoutRate;
        this.loadBalancingWeight = loadBalancingWeight;
        this.totalLoadBalancingLoss = 0.0f;
        
        init();
    }
    
    /**
     * 创建中等规模MoE-GPT模型的工厂方法
     * 
     * @param name 模型名称
     * @param vocabSize 词汇表大小
     * @return MoE-GPT模型实例
     */
    public static MoEGPTModel createMediumModel(String name, int vocabSize) {
        return new MoEGPTModel(
            name,
            vocabSize,    // 词汇表大小
            512,          // 嵌入维度
            8,            // 8层Transformer
            8,            // 8个注意力头
            8,            // 8个专家
            2,            // Top-2专家选择
            2048,         // 专家隐藏维度
            1024,         // 最大序列长度
            0.1,          // Dropout比率
            0.01          // 负载均衡权重
        );
    }
    
    /**
     * 创建小规模MoE-GPT模型的工厂方法
     * 适合实验和快速原型
     * 
     * @param name 模型名称
     * @param vocabSize 词汇表大小
     * @return MoE-GPT模型实例
     */
    public static MoEGPTModel createSmallModel(String name, int vocabSize) {
        return new MoEGPTModel(
            name,
            vocabSize,    // 词汇表大小
            256,          // 嵌入维度
            6,            // 6层Transformer
            8,            // 8个注意力头
            4,            // 4个专家
            2,            // Top-2专家选择
            1024,         // 专家隐藏维度
            512,          // 最大序列长度
            0.1,          // Dropout比率
            0.01          // 负载均衡权重
        );
    }
    
    /**
     * 创建微型MoE-GPT模型的工厂方法
     * 用于调试和概念验证
     * 
     * @param name 模型名称
     * @param vocabSize 词汇表大小
     * @return MoE-GPT模型实例
     */
    public static MoEGPTModel createTinyModel(String name, int vocabSize) {
        return new MoEGPTModel(
            name,
            vocabSize,    // 词汇表大小
            128,          // 嵌入维度
            4,            // 4层Transformer
            4,            // 4个注意力头
            2,            // 2个专家
            1,            // Top-1专家选择
            512,          // 专家隐藏维度
            256,          // 最大序列长度
            0.1,          // Dropout比率
            0.01          // 负载均衡权重
        );
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            System.out.println("正在初始化MoE-GPT模型:");
            System.out.println("  词汇表大小: " + vocabSize);
            System.out.println("  模型维度: " + dModel);
            System.out.println("  层数: " + numLayers);
            System.out.println("  注意力头数: " + numHeads);
            System.out.println("  专家数量: " + numExperts);
            System.out.println("  Top-K: " + topK);
            System.out.println("  最大序列长度: " + maxSeqLength);
            
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
            
            // 2. 初始化MoE Transformer块
            moeTransformerBlocks = new ArrayList<>();
            for (int i = 0; i < numLayers; i++) {
                MoETransformerBlock block = new MoETransformerBlock(
                    name + "_moe_block_" + i,
                    dModel,
                    numHeads,
                    numExperts,
                    topK,
                    expertHiddenDim,
                    dropoutRate,
                    loadBalancingWeight
                );
                moeTransformerBlocks.add(block);
                addLayer(block);
            }
            
            // 3. 初始化最终层归一化
            finalLayerNorm = new LayerNorm(name + "_final_ln", dModel);
            addLayer(finalLayerNorm);
            
            // 4. 初始化输出头
            outputHead = new GPT2OutputHead(name + "_output_head", dModel, vocabSize);
            addLayer(outputHead);
            
            alreadyInit = true;
            System.out.println("MoE-GPT模型初始化完成。");
            System.out.println("总参数量: " + getParameterCount());
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable tokenIds = inputs[0];  // shape: (batch_size, seq_len)
        
        // 验证输入形状
        NdArray inputData = tokenIds.getValue();
        if (inputData.shape.dimension.length != 2) {
            throw new IllegalArgumentException("输入必须是2D张量 (batch_size, seq_len)");
        }
        
        int seqLen = inputData.shape.dimension[1];
        if (seqLen > maxSeqLength) {
            throw new IllegalArgumentException(
                String.format("输入序列长度 %d 超过最大长度 %d", seqLen, maxSeqLength)
            );
        }
        
        // 1. Token嵌入 + 位置嵌入
        Variable x = tokenEmbedding.layerForward(tokenIds);
        
        // 2. 通过所有MoE Transformer块
        for (MoETransformerBlock block : moeTransformerBlocks) {
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
     * 
     * @param tokenIds 输入token序列
     * @return 下一个token的概率分布
     */
    public Variable generate(Variable tokenIds) {
        return layerForward(tokenIds);
    }
    
    /**
     * 预测下一个token（贪心解码）
     * 
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
     * 计算总的负载均衡损失
     * 这个损失应该添加到训练损失中以鼓励专家的均匀使用
     * 
     * @return 总负载均衡损失
     */
    public float computeTotalLoadBalancingLoss() {
        float totalLoss = 0.0f;
        for (MoETransformerBlock block : moeTransformerBlocks) {
            totalLoss += block.getLoadBalancingLoss();
        }
        this.totalLoadBalancingLoss = totalLoss;
        return totalLoss;
    }
    
    /**
     * 重置所有MoE块的专家使用统计
     */
    public void resetAllExpertStatistics() {
        for (MoETransformerBlock block : moeTransformerBlocks) {
            block.resetExpertStatistics();
        }
        this.totalLoadBalancingLoss = 0.0f;
    }
    
    /**
     * 打印所有层的专家使用统计
     */
    public void printAllExpertStatistics() {
        System.out.println("\n=== MoE-GPT模型专家使用统计 ===");
        System.out.println("模型名称: " + name);
        System.out.println("总负载均衡损失: " + String.format("%.6f", computeTotalLoadBalancingLoss()));
        System.out.println();
        
        for (int i = 0; i < moeTransformerBlocks.size(); i++) {
            System.out.println("--- 第" + (i + 1) + "层 ---");
            moeTransformerBlocks.get(i).printExpertStatistics();
            System.out.println();
        }
        System.out.println("================================");
    }
    
    /**
     * 获取各层专家使用率的汇总统计
     * 
     * @return 每层的专家使用率数组
     */
    public List<float[]> getAllLayersExpertUsageRates() {
        List<float[]> allUsageRates = new ArrayList<>();
        for (MoETransformerBlock block : moeTransformerBlocks) {
            allUsageRates.add(block.getExpertUsageRates());
        }
        return allUsageRates;
    }
    
    /**
     * 计算模型的总参数量
     * 
     * @return 模型总参数量
     */
    public long getParameterCount() {
        long totalParams = 0;
        
        // Token嵌入参数
        totalParams += (long) vocabSize * dModel;
        
        // 位置嵌入参数
        totalParams += (long) maxSeqLength * dModel;
        
        // 所有MoE Transformer块的参数
        for (MoETransformerBlock block : moeTransformerBlocks) {
            totalParams += block.getParameterCount();
        }
        
        // 最终层归一化参数
        totalParams += 2L * dModel;
        
        // 输出头参数
        totalParams += (long) dModel * vocabSize;
        
        return totalParams;
    }
    
    /**
     * 计算有效参数量（考虑MoE的稀疏性）
     * 
     * @return 每次前向传播实际使用的参数量
     */
    public long getActiveParameterCount() {
        long activeParams = 0;
        
        // Token嵌入和位置嵌入（总是激活）
        activeParams += (long) vocabSize * dModel;
        activeParams += (long) maxSeqLength * dModel;
        
        // 每个MoE块的激活参数
        for (int i = 0; i < numLayers; i++) {
            // LayerNorm参数（总是激活）
            activeParams += 2L * 2 * dModel;
            
            // MultiHeadAttention参数（总是激活）
            activeParams += 4L * (dModel * dModel + dModel);
            
            // MoE层：只有topK个专家被激活
            // 门控网络参数（总是激活）
            activeParams += (long) dModel * numExperts + numExperts;
            
            // 激活的专家参数
            long singleExpertParams = 2L * dModel * expertHiddenDim + expertHiddenDim + dModel;
            activeParams += singleExpertParams * topK;
        }
        
        // 最终层归一化和输出头（总是激活）
        activeParams += 2L * dModel;
        activeParams += (long) dModel * vocabSize;
        
        return activeParams;
    }
    
    /**
     * 打印模型信息
     */
    public void printModelInfo() {
        System.out.println("=== MoE-GPT模型信息 ===");
        System.out.println("模型名称: " + name);
        System.out.println("词汇表大小: " + vocabSize);
        System.out.println("模型维度: " + dModel);
        System.out.println("层数: " + numLayers);
        System.out.println("注意力头数: " + numHeads);
        System.out.println("专家数量: " + numExperts);
        System.out.println("Top-K选择: " + topK);
        System.out.println("专家隐藏维度: " + expertHiddenDim);
        System.out.println("最大序列长度: " + maxSeqLength);
        System.out.println("Dropout比率: " + dropoutRate);
        System.out.println("负载均衡权重: " + loadBalancingWeight);
        System.out.println("总参数量: " + getParameterCount());
        System.out.println("激活参数量: " + getActiveParameterCount());
        System.out.printf("参数效率: %.2f%% (激活/总计)%n", 
            100.0 * getActiveParameterCount() / getParameterCount());
        System.out.println("====================");
    }
    
    // Getters
    public GPT2TokenEmbedding getTokenEmbedding() { return tokenEmbedding; }
    public List<MoETransformerBlock> getMoeTransformerBlocks() { return moeTransformerBlocks; }
    public LayerNorm getFinalLayerNorm() { return finalLayerNorm; }
    public GPT2OutputHead getOutputHead() { return outputHead; }
    public int getVocabSize() { return vocabSize; }
    public int getDModel() { return dModel; }
    public int getNumLayers() { return numLayers; }
    public int getNumHeads() { return numHeads; }
    public int getNumExperts() { return numExperts; }
    public int getTopK() { return topK; }
    public int getExpertHiddenDim() { return expertHiddenDim; }
    public int getMaxSeqLength() { return maxSeqLength; }
    public double getDropoutRate() { return dropoutRate; }
    public double getLoadBalancingWeight() { return loadBalancingWeight; }
    public float getTotalLoadBalancingLoss() { return totalLoadBalancingLoss; }
}