package io.leavesfly.tinydl.modality.nlp.layer;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.func.math.Exp;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Layer;
import io.leavesfly.tinydl.nnet.Parameter;
import io.leavesfly.tinydl.nnet.layer.dnn.LinearLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * MoE门控网络实现
 * 
 * 门控网络负责计算每个专家的权重，决定对于给定输入应该使用哪些专家。
 * 门控网络通常使用softmax函数来生成归一化的权重分布。
 * 
 * 公式：G(x) = softmax(xW_g + b_g)
 * 其中G(x)是一个长度为num_experts的概率分布向量
 * 
 * @author leavesfly
 * @version 0.01
 */
public class MoEGatingNetwork extends Layer {
    
    /**
     * 门控网络的线性层
     */
    private LinearLayer gatingLinear;
    
    /**
     * 输入维度
     */
    private int inputDim;
    
    /**
     * 专家数量
     */
    private int numExperts;
    
    /**
     * Top-K选择的专家数量（稀疏化参数）
     */
    private int topK;
    
    /**
     * 噪声因子，用于负载均衡
     */
    private double noiseFactor;
    
    /**
     * 构造MoE门控网络
     * 
     * @param name 层名称
     * @param inputDim 输入维度
     * @param numExperts 专家数量
     * @param topK 选择的top-K专家数量
     * @param noiseFactor 噪声因子
     */
    public MoEGatingNetwork(String name, int inputDim, int numExperts, int topK, double noiseFactor) {
        super(name, new Shape(-1, -1, inputDim), new Shape(-1, -1, numExperts));
        this.inputDim = inputDim;
        this.numExperts = numExperts;
        this.topK = Math.min(topK, numExperts); // 确保topK不超过专家数量
        this.noiseFactor = noiseFactor;
        init();
    }
    
    /**
     * 默认构造函数，topK=2，无噪声
     */
    public MoEGatingNetwork(String name, int inputDim, int numExperts) {
        this(name, inputDim, numExperts, 2, 0.0);
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化门控线性层：将输入映射到专家数量的维度
            gatingLinear = new LinearLayer(name + "_gating_linear", inputDim, numExperts, true);
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
        
        // 将三维输入重塑为二维以进行矩阵乘法
        NdArray inputReshaped = inputData.reshape(new Shape(batchSize * seqLen, inputDim));
        Variable reshapedInput = new Variable(inputReshaped);
        
        // 通过门控线性层
        Variable gatingLogits = gatingLinear.layerForward(reshapedInput);
        
        // 添加噪声（用于训练时的负载均衡）
        if (noiseFactor > 0.0) {
            gatingLogits = addGatingNoise(gatingLogits);
        }
        
        // 应用softmax获得权重分布
        Variable gatingWeights = applySoftmax(gatingLogits);
        
        // Top-K专家选择（可选的稀疏化）
        if (topK < numExperts) {
            gatingWeights = applyTopKSparsity(gatingWeights);
        }
        
        // 重塑回三维
        NdArray weightsReshaped = gatingWeights.getValue().reshape(new Shape(batchSize, seqLen, numExperts));
        
        return new Variable(weightsReshaped);
    }
    
    /**
     * 为门控logits添加噪声，用于负载均衡
     * 
     * @param logits 门控logits
     * @return 添加噪声后的logits
     */
    private Variable addGatingNoise(Variable logits) {
        // 简化版本：添加高斯噪声
        NdArray logitsData = logits.getValue();
        NdArray noise = NdArray.likeRandomN(logitsData.shape).mulNum(noiseFactor);
        NdArray noisyLogits = logitsData.add(noise);
        return new Variable(noisyLogits);
    }
    
    /**
     * 应用softmax函数
     * 
     * @param logits 输入logits
     * @return softmax后的概率分布
     */
    private Variable applySoftmax(Variable logits) {
        // 简化版本的softmax实现
        NdArray logitsData = logits.getValue();
        int totalTokens = logitsData.shape.dimension[0];
        
        NdArray softmaxResult = NdArray.zeros(logitsData.shape);
        
        for (int i = 0; i < totalTokens; i++) {
            // 找到最大值以避免数值溢出
            float maxLogit = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < numExperts; j++) {
                float logit = logitsData.get(i, j);
                if (logit > maxLogit) {
                    maxLogit = logit;
                }
            }
            
            // 计算exp和sum
            float sumExp = 0.0f;
            float[] expValues = new float[numExperts];
            for (int j = 0; j < numExperts; j++) {
                expValues[j] = (float) Math.exp(logitsData.get(i, j) - maxLogit);
                sumExp += expValues[j];
            }
            
            // 归一化
            for (int j = 0; j < numExperts; j++) {
                softmaxResult.set(expValues[j] / sumExp, i, j);
            }
        }
        
        return new Variable(softmaxResult);
    }
    
    /**
     * 应用Top-K稀疏化，只保留权重最大的K个专家
     * 
     * @param weights 专家权重
     * @return 稀疏化后的权重
     */
    private Variable applyTopKSparsity(Variable weights) {
        NdArray weightsData = weights.getValue();
        int totalTokens = weightsData.shape.dimension[0];
        
        NdArray sparsedWeights = NdArray.zeros(weightsData.shape);
        
        for (int i = 0; i < totalTokens; i++) {
            // 找到topK个最大权重的专家
            List<Integer> topExperts = findTopKExperts(weightsData, i, topK);
            
            // 重新归一化topK专家的权重
            float sumTopK = 0.0f;
            for (int expertIdx : topExperts) {
                sumTopK += weightsData.get(i, expertIdx);
            }
            
            // 设置topK专家的归一化权重
            for (int expertIdx : topExperts) {
                float normalizedWeight = weightsData.get(i, expertIdx) / sumTopK;
                sparsedWeights.set(normalizedWeight, i, expertIdx);
            }
        }
        
        return new Variable(sparsedWeights);
    }
    
    /**
     * 找到权重最大的K个专家
     * 
     * @param weights 权重数组
     * @param tokenIdx token索引
     * @param k 要选择的专家数量
     * @return top-K专家的索引列表
     */
    private List<Integer> findTopKExperts(NdArray weights, int tokenIdx, int k) {
        List<Integer> allExperts = new ArrayList<>();
        for (int i = 0; i < numExperts; i++) {
            allExperts.add(i);
        }
        
        // 按权重降序排序
        allExperts.sort((a, b) -> Float.compare(weights.get(tokenIdx, b), weights.get(tokenIdx, a)));
        
        // 返回前K个
        return allExperts.subList(0, k);
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
    /**
     * 获取门控线性层
     */
    public LinearLayer getGatingLinear() {
        return gatingLinear;
    }
    
    /**
     * 获取输入维度
     */
    public int getInputDim() {
        return inputDim;
    }
    
    /**
     * 获取专家数量
     */
    public int getNumExperts() {
        return numExperts;
    }
    
    /**
     * 获取Top-K参数
     */
    public int getTopK() {
        return topK;
    }
    
    /**
     * 获取噪声因子
     */
    public double getNoiseFactor() {
        return noiseFactor;
    }
    
    /**
     * 设置Top-K参数
     */
    public void setTopK(int topK) {
        this.topK = Math.min(topK, numExperts);
    }
    
    /**
     * 设置噪声因子
     */
    public void setNoiseFactor(double noiseFactor) {
        this.noiseFactor = noiseFactor;
    }
}