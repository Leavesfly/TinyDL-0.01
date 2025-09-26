package io.leavesfly.tinydl.modality.nlp.layer;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Layer;

import java.util.ArrayList;
import java.util.List;

/**
 * Mixture of Experts (MoE) 层实现
 * 
 * MoE层是一种高效的神经网络架构，通过组合多个专家网络来增加模型容量。
 * 核心思想是对于不同的输入，激活不同的专家子集，从而实现专业化处理。
 * 
 * MoE层的工作流程：
 * 1. 门控网络（Gating Network）计算每个专家的权重
 * 2. 根据Top-K策略选择最相关的专家
 * 3. 将输入路由到选中的专家进行处理
 * 4. 根据权重对专家输出进行加权求和
 * 
 * 公式：
 * MoE(x) = Σ(G(x)_i * E_i(x)) for i in Top-K experts
 * 其中 G(x) 是门控权重，E_i(x) 是第i个专家的输出
 * 
 * @author leavesfly
 * @version 0.01
 */
public class MoELayer extends Layer {
    
    /**
     * 门控网络
     */
    private MoEGatingNetwork gatingNetwork;
    
    /**
     * 专家网络列表
     */
    private List<MoEExpertNetwork> experts;
    
    /**
     * 输入维度
     */
    private int inputDim;
    
    /**
     * 输出维度
     */
    private int outputDim;
    
    /**
     * 专家数量
     */
    private int numExperts;
    
    /**
     * Top-K选择的专家数量
     */
    private int topK;
    
    /**
     * 专家隐藏层维度
     */
    private int expertHiddenDim;
    
    /**
     * 负载均衡损失权重
     */
    private double loadBalancingWeight;
    
    /**
     * 是否启用负载均衡
     */
    private boolean enableLoadBalancing;
    
    /**
     * 专家使用统计（用于负载均衡分析）
     */
    private int[] expertUsageCount;
    
    /**
     * 总处理的token数量
     */
    private long totalTokensProcessed;
    
    /**
     * 构造MoE层
     * 
     * @param name 层名称
     * @param inputDim 输入维度
     * @param outputDim 输出维度  
     * @param numExperts 专家数量
     * @param topK 选择的top-K专家数量
     * @param expertHiddenDim 专家隐藏层维度
     * @param loadBalancingWeight 负载均衡损失权重
     */
    public MoELayer(String name, int inputDim, int outputDim, int numExperts, 
                    int topK, int expertHiddenDim, double loadBalancingWeight) {
        super(name, new Shape(-1, -1, inputDim), new Shape(-1, -1, outputDim));
        this.inputDim = inputDim;
        this.outputDim = outputDim;
        this.numExperts = numExperts;
        this.topK = Math.min(topK, numExperts);
        this.expertHiddenDim = expertHiddenDim;
        this.loadBalancingWeight = loadBalancingWeight;
        this.enableLoadBalancing = loadBalancingWeight > 0.0;
        this.expertUsageCount = new int[numExperts];
        this.totalTokensProcessed = 0;
        init();
    }
    
    /**
     * 简化构造函数，使用默认参数
     * 
     * @param name 层名称
     * @param modelDim 模型维度（输入输出维度相同）
     * @param numExperts 专家数量
     */
    public MoELayer(String name, int modelDim, int numExperts) {
        this(name, modelDim, modelDim, numExperts, 2, modelDim * 4, 0.01);
    }
    
    /**
     * 带Top-K参数的构造函数
     * 
     * @param name 层名称
     * @param modelDim 模型维度
     * @param numExperts 专家数量
     * @param topK Top-K专家数量
     */
    public MoELayer(String name, int modelDim, int numExperts, int topK) {
        this(name, modelDim, modelDim, numExperts, topK, modelDim * 4, 0.01);
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化门控网络
            gatingNetwork = new MoEGatingNetwork(
                name + "_gating", 
                inputDim, 
                numExperts, 
                topK, 
                0.1  // 噪声因子
            );
            
            // 初始化专家网络
            experts = new ArrayList<>();
            for (int i = 0; i < numExperts; i++) {
                MoEExpertNetwork expert = new MoEExpertNetwork(
                    name + "_expert_" + i,
                    i,
                    inputDim,
                    expertHiddenDim,
                    outputDim,
                    0.1  // Dropout率
                );
                experts.add(expert);
            }
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0]; // shape: (batch_size, seq_len, input_dim)
        
        // 获取输入形状
        NdArray inputData = input.getValue();
        int batchSize = inputData.shape.dimension[0];
        int seqLen = inputData.shape.dimension[1];
        int totalTokens = batchSize * seqLen;
        
        // 1. 通过门控网络计算专家权重
        Variable gatingWeights = gatingNetwork.layerForward(input); // shape: (batch_size, seq_len, num_experts)
        
        // 2. 获取所有专家的输出
        List<Variable> expertOutputs = new ArrayList<>();
        for (MoEExpertNetwork expert : experts) {
            Variable expertOutput = expert.layerForward(input); // shape: (batch_size, seq_len, output_dim)
            expertOutputs.add(expertOutput);
        }
        
        // 3. 根据门控权重对专家输出进行加权求和
        Variable finalOutput = computeWeightedSum(expertOutputs, gatingWeights, batchSize, seqLen);
        
        // 4. 更新专家使用统计（用于负载均衡分析）
        if (enableLoadBalancing) {
            updateExpertUsageStatistics(gatingWeights);
        }
        
        totalTokensProcessed += totalTokens;
        
        return finalOutput;
    }
    
    /**
     * 计算专家输出的加权求和
     * 
     * @param expertOutputs 所有专家的输出列表
     * @param gatingWeights 门控权重
     * @param batchSize 批次大小
     * @param seqLen 序列长度
     * @return 加权求和后的最终输出
     */
    private Variable computeWeightedSum(List<Variable> expertOutputs, Variable gatingWeights, 
                                       int batchSize, int seqLen) {
        
        NdArray weightsData = gatingWeights.getValue();
        NdArray result = NdArray.zeros(new Shape(batchSize, seqLen, outputDim));
        
        // 对每个token位置进行加权求和
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                // 当前token位置的输出向量
                for (int d = 0; d < outputDim; d++) {
                    float weightedSum = 0.0f;
                    
                    // 对所有专家进行加权求和
                    for (int e = 0; e < numExperts; e++) {
                        float weight = weightsData.get(b, s, e);
                        if (weight > 1e-8f) { // 只考虑权重大于阈值的专家
                            float expertValue = expertOutputs.get(e).getValue().get(b, s, d);
                            weightedSum += weight * expertValue;
                        }
                    }
                    
                    result.set(weightedSum, b, s, d);
                }
            }
        }
        
        return new Variable(result);
    }
    
    /**
     * 更新专家使用统计信息
     * 
     * @param gatingWeights 门控权重
     */
    private void updateExpertUsageStatistics(Variable gatingWeights) {
        NdArray weightsData = gatingWeights.getValue();
        int batchSize = weightsData.shape.dimension[0];
        int seqLen = weightsData.shape.dimension[1];
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int e = 0; e < numExperts; e++) {
                    float weight = weightsData.get(b, s, e);
                    if (weight > 1e-8f) { // 专家被使用
                        expertUsageCount[e]++;
                    }
                }
            }
        }
    }
    
    /**
     * 计算负载均衡损失
     * 负载均衡损失鼓励专家使用的均匀分布，避免某些专家被过度使用
     * 
     * @return 负载均衡损失值
     */
    public float computeLoadBalancingLoss() {
        if (!enableLoadBalancing || totalTokensProcessed == 0) {
            return 0.0f;
        }
        
        // 计算专家使用频率
        float[] usageFrequencies = new float[numExperts];
        for (int i = 0; i < numExperts; i++) {
            usageFrequencies[i] = (float) expertUsageCount[i] / totalTokensProcessed;
        }
        
        // 计算方差作为负载均衡损失
        float mean = 1.0f / numExperts; // 理想的均匀分布
        float variance = 0.0f;
        for (float freq : usageFrequencies) {
            variance += (freq - mean) * (freq - mean);
        }
        variance /= numExperts;
        
        return (float) (loadBalancingWeight * variance);
    }
    
    /**
     * 重置专家使用统计
     */
    public void resetUsageStatistics() {
        for (int i = 0; i < numExperts; i++) {
            expertUsageCount[i] = 0;
        }
        totalTokensProcessed = 0;
    }
    
    /**
     * 获取专家使用率统计信息
     * 
     * @return 专家使用率数组
     */
    public float[] getExpertUsageRates() {
        if (totalTokensProcessed == 0) {
            return new float[numExperts];
        }
        
        float[] usageRates = new float[numExperts];
        for (int i = 0; i < numExperts; i++) {
            usageRates[i] = (float) expertUsageCount[i] / totalTokensProcessed;
        }
        return usageRates;
    }
    
    /**
     * 打印专家使用统计信息
     */
    public void printExpertUsageStatistics() {
        System.out.println("=== MoE专家使用统计 ===");
        System.out.println("总处理Token数: " + totalTokensProcessed);
        System.out.println("负载均衡损失: " + String.format("%.6f", computeLoadBalancingLoss()));
        
        float[] usageRates = getExpertUsageRates();
        for (int i = 0; i < numExperts; i++) {
            System.out.printf("专家%d: 使用率=%.4f%%, 使用次数=%d%n", 
                i, usageRates[i] * 100, expertUsageCount[i]);
        }
        System.out.println("====================");
    }
    
    /**
     * 计算MoE层的总参数量
     * 
     * @return 总参数量
     */
    public long getParameterCount() {
        long totalParams = 0;
        
        // 门控网络参数
        totalParams += (long) inputDim * numExperts + numExperts; // 线性层参数
        
        // 所有专家网络参数
        for (MoEExpertNetwork expert : experts) {
            totalParams += expert.getParameterCount();
        }
        
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
    
    // Getters
    public MoEGatingNetwork getGatingNetwork() {
        return gatingNetwork;
    }
    
    public List<MoEExpertNetwork> getExperts() {
        return experts;
    }
    
    public int getInputDim() {
        return inputDim;
    }
    
    public int getOutputDim() {
        return outputDim;
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
    
    public double getLoadBalancingWeight() {
        return loadBalancingWeight;
    }
    
    public boolean isLoadBalancingEnabled() {
        return enableLoadBalancing;
    }
    
    public long getTotalTokensProcessed() {
        return totalTokensProcessed;
    }
    
    // Setters
    public void setLoadBalancingWeight(double weight) {
        this.loadBalancingWeight = weight;
        this.enableLoadBalancing = weight > 0.0;
    }
    
    public void setTopK(int topK) {
        this.topK = Math.min(topK, numExperts);
        if (gatingNetwork != null) {
            gatingNetwork.setTopK(topK);
        }
    }
}