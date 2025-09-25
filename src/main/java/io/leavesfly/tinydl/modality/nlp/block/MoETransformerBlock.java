package io.leavesfly.tinydl.modality.nlp.block;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.modality.nlp.layer.MoELayer;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Layer;
import io.leavesfly.tinydl.nnet.layer.transformer.LayerNorm;
import io.leavesfly.tinydl.nnet.layer.transformer.MultiHeadAttention;

import java.util.ArrayList;
import java.util.List;

/**
 * 支持MoE的Transformer Block实现
 * 
 * 这个Block替换传统的FeedForward层为MoE层，从而大幅增加模型容量
 * 同时保持计算效率。结构如下：
 * 
 * 1. 层归一化（前置）
 * 2. 带掩码的多头自注意力
 * 3. 残差连接
 * 4. 层归一化（前置）
 * 5. MoE层（替换传统FeedForward）
 * 6. 残差连接
 * 
 * MoE层通过专家路由机制，让不同的输入激活不同的专家子集，
 * 实现了模型容量的大幅提升而不增加太多计算开销。
 * 
 * @author leavesfly
 * @version 0.01
 */
public class MoETransformerBlock extends Layer {
    
    /**
     * 第一个层归一化（注意力前）
     */
    private LayerNorm layerNorm1;
    
    /**
     * 带掩码的多头自注意力
     */
    private MultiHeadAttention attention;
    
    /**
     * 第二个层归一化（MoE前）
     */
    private LayerNorm layerNorm2;
    
    /**
     * MoE层（替换传统FeedForward）
     */
    private MoELayer moeLayer;
    
    /**
     * 模型维度
     */
    private int dModel;
    
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
     * Dropout比率
     */
    private double dropoutRate;
    
    /**
     * 负载均衡权重
     */
    private double loadBalancingWeight;
    
    /**
     * 是否启用残差连接
     */
    private boolean enableResidualConnection;
    
    /**
     * 构造支持MoE的Transformer Block
     * 
     * @param name 块名称
     * @param dModel 模型维度
     * @param numHeads 注意力头数
     * @param numExperts 专家数量
     * @param topK Top-K专家选择数量
     * @param expertHiddenDim 专家隐藏层维度
     * @param dropoutRate Dropout比率
     * @param loadBalancingWeight 负载均衡权重
     */
    public MoETransformerBlock(String name, int dModel, int numHeads, int numExperts, 
                              int topK, int expertHiddenDim, double dropoutRate, 
                              double loadBalancingWeight) {
        super(name, new Shape(-1, -1, dModel), new Shape(-1, -1, dModel));
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.numExperts = numExperts;
        this.topK = topK;
        this.expertHiddenDim = expertHiddenDim;
        this.dropoutRate = dropoutRate;
        this.loadBalancingWeight = loadBalancingWeight;
        this.enableResidualConnection = true;
        init();
    }
    
    /**
     * 简化构造函数，使用默认参数
     * 
     * @param name 块名称
     * @param dModel 模型维度
     * @param numHeads 注意力头数
     * @param numExperts 专家数量
     */
    public MoETransformerBlock(String name, int dModel, int numHeads, int numExperts) {
        this(name, dModel, numHeads, numExperts, 2, dModel * 4, 0.1, 0.01);
    }
    
    /**
     * 带Top-K参数的构造函数
     * 
     * @param name 块名称
     * @param dModel 模型维度
     * @param numHeads 注意力头数
     * @param numExperts 专家数量
     * @param topK Top-K专家选择数量
     */
    public MoETransformerBlock(String name, int dModel, int numHeads, int numExperts, int topK) {
        this(name, dModel, numHeads, numExperts, topK, dModel * 4, 0.1, 0.01);
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化第一个层归一化
            layerNorm1 = new LayerNorm(name + "_ln1", dModel);
            
            // 初始化多头注意力（带掩码，用于GPT式的自回归生成）
            attention = new MultiHeadAttention(name + "_attention", dModel, numHeads, true);
            
            // 初始化第二个层归一化
            layerNorm2 = new LayerNorm(name + "_ln2", dModel);
            
            // 初始化MoE层
            moeLayer = new MoELayer(
                name + "_moe",
                dModel,           // 输入维度
                dModel,           // 输出维度
                numExperts,       // 专家数量
                topK,             // Top-K选择
                expertHiddenDim,  // 专家隐藏维度
                loadBalancingWeight // 负载均衡权重
            );
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable x = inputs[0]; // shape: (batch_size, seq_len, d_model)
        
        // Pre-LayerNorm架构
        
        // 1. 第一个子层：LayerNorm + Multi-Head Attention + Residual
        Variable norm1Output = layerNorm1.layerForward(x);
        Variable attentionOutput = attention.layerForward(norm1Output, norm1Output, norm1Output);
        
        // 应用Dropout到注意力输出
        if (dropoutRate > 0.0) {
            attentionOutput = applyDropout(attentionOutput, "attention");
        }
        
        // 残差连接
        Variable residual1 = enableResidualConnection ? 
            addResidualConnection(x, attentionOutput) : attentionOutput;
        
        // 2. 第二个子层：LayerNorm + MoE + Residual
        Variable norm2Output = layerNorm2.layerForward(residual1);
        Variable moeOutput = moeLayer.layerForward(norm2Output);
        
        // 应用Dropout到MoE输出
        if (dropoutRate > 0.0) {
            moeOutput = applyDropout(moeOutput, "moe");
        }
        
        // 残差连接
        Variable residual2 = enableResidualConnection ? 
            addResidualConnection(residual1, moeOutput) : moeOutput;
        
        return residual2;
    }
    
    /**
     * 添加残差连接
     * 
     * @param input 输入张量
     * @param output 子层输出张量
     * @return 残差连接结果
     */
    private Variable addResidualConnection(Variable input, Variable output) {
        return input.add(output);
    }
    
    /**
     * 应用Dropout正则化
     * 简化版本的Dropout实现
     * 
     * @param input 输入变量
     * @param layerType 层类型（用于调试）
     * @return 应用Dropout后的变量
     */
    private Variable applyDropout(Variable input, String layerType) {
        if (dropoutRate <= 0.0) {
            return input;
        }
        
        // 简化的Dropout实现
        NdArray inputData = input.getValue();
        NdArray droppedData = NdArray.zeros(inputData.shape);
        
        for (int i = 0; i < inputData.buffer.length; i++) {
            if (Math.random() > dropoutRate) {
                // 保留该神经元，并进行缩放补偿
                droppedData.buffer[i] = inputData.buffer[i] / (float)(1.0 - dropoutRate);
            }
            // 否则保持为0（已经在zeros中初始化）
        }
        
        return new Variable(droppedData);
    }
    
    /**
     * 获取负载均衡损失
     * 这个损失可以添加到总的训练损失中，以鼓励专家的均匀使用
     * 
     * @return 负载均衡损失值
     */
    public float getLoadBalancingLoss() {
        return moeLayer.computeLoadBalancingLoss();
    }
    
    /**
     * 重置MoE专家使用统计
     */
    public void resetExpertStatistics() {
        moeLayer.resetUsageStatistics();
    }
    
    /**
     * 打印专家使用统计信息
     */
    public void printExpertStatistics() {
        System.out.println("=== " + name + " MoE统计 ===");
        moeLayer.printExpertUsageStatistics();
    }
    
    /**
     * 获取专家使用率
     * 
     * @return 专家使用率数组
     */
    public float[] getExpertUsageRates() {
        return moeLayer.getExpertUsageRates();
    }
    
    /**
     * 计算Block的总参数量
     * 
     * @return 总参数量
     */
    public long getParameterCount() {
        long totalParams = 0;
        
        // LayerNorm参数 (2个LayerNorm，每个有2*dModel个参数)
        totalParams += 2L * 2 * dModel;
        
        // MultiHeadAttention参数
        // Q, K, V, O 四个线性层：4 * (dModel * dModel + dModel)
        totalParams += 4L * (dModel * dModel + dModel);
        
        // MoE层参数
        totalParams += moeLayer.getParameterCount();
        
        return totalParams;
    }
    
    @Override
    public NdArray forward(NdArray... inputs) {
        return layerForward(new Variable(inputs[0])).getValue();
    }
    
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 简化版本的反向传播
        List<NdArray> result = new ArrayList<>();
        result.add(yGrad);
        return result;
    }
    
    @Override
    public int requireInputNum() {
        return 1;
    }
    
    // Getters and Setters
    public LayerNorm getLayerNorm1() {
        return layerNorm1;
    }
    
    public MultiHeadAttention getAttention() {
        return attention;
    }
    
    public LayerNorm getLayerNorm2() {
        return layerNorm2;
    }
    
    public MoELayer getMoeLayer() {
        return moeLayer;
    }
    
    public int getDModel() {
        return dModel;
    }
    
    public int getNumHeads() {
        return numHeads;
    }
    
    public int getNumExperts() {
        return numExperts;
    }
    
    public int getTopK() {
        return topK;
    }
    
    public int getExpertHiddenDim() {
        return expertHiddenDim;
    }
    
    public double getDropoutRate() {
        return dropoutRate;
    }
    
    public double getLoadBalancingWeight() {
        return loadBalancingWeight;
    }
    
    public boolean isResidualConnectionEnabled() {
        return enableResidualConnection;
    }
    
    public void setDropoutRate(double dropoutRate) {
        this.dropoutRate = Math.max(0.0, Math.min(1.0, dropoutRate));
    }
    
    public void setLoadBalancingWeight(double weight) {
        this.loadBalancingWeight = weight;
        if (moeLayer != null) {
            moeLayer.setLoadBalancingWeight(weight);
        }
    }
    
    public void setTopK(int topK) {
        this.topK = Math.min(topK, numExperts);
        if (moeLayer != null) {
            moeLayer.setTopK(topK);
        }
    }
    
    public void setResidualConnectionEnabled(boolean enabled) {
        this.enableResidualConnection = enabled;
    }
}