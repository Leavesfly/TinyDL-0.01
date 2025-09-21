package io.leavesfly.tinydl.nnet.layer.transformer;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Layer;
import io.leavesfly.tinydl.nnet.Parameter;

import java.util.ArrayList;
import java.util.List;

/**
 * 层归一化（Layer Normalization）实现
 * 
 * 层归一化对每个样本的特征维度进行归一化，公式如下：
 * LayerNorm(x) = γ * (x - μ) / σ + β
 * 
 * 其中：
 * - μ 是均值
 * - σ 是标准差
 * - γ 是学习的缩放参数
 * - β 是学习的偏移参数
 */
public class LayerNorm extends Layer {
    
    private Parameter gamma; // 缩放参数
    private Parameter beta;  // 偏移参数
    private double epsilon;  // 防止除零的小常数
    private int normalizedShape; // 归一化的维度大小
    
    /**
     * 构造层归一化层
     * 
     * @param name 层名称
     * @param normalizedShape 归一化的特征维度大小
     * @param epsilon 防止除零的小常数
     */
    public LayerNorm(String name, int normalizedShape, double epsilon) {
        super(name, new Shape(-1, -1, normalizedShape), new Shape(-1, -1, normalizedShape));
        this.normalizedShape = normalizedShape;
        this.epsilon = epsilon;
        init();
    }
    
    public LayerNorm(String name, int normalizedShape) {
        this(name, normalizedShape, 1e-6);
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化γ为1，β为0
            gamma = new Parameter(NdArray.ones(new Shape(1, normalizedShape)));
            gamma.setName(name + "_gamma");
            addParam(gamma.getName(), gamma);
            
            beta = new Parameter(NdArray.zeros(new Shape(1, normalizedShape)));
            beta.setName(name + "_beta");
            addParam(beta.getName(), beta);
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];
        NdArray inputData = input.getValue();
        
        int[] shape = inputData.shape.dimension;
        int batchSize = shape[0];
        int seqLen = shape[1];
        int featureDim = shape[2];
        
        if (featureDim != normalizedShape) {
            throw new IllegalArgumentException("Input feature dimension doesn't match normalized shape");
        }
        
        NdArray output = new NdArray(new Shape(batchSize, seqLen, featureDim));
        
        // 对每个位置进行层归一化
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                // 计算均值
                float sum = 0.0f;
                for (int f = 0; f < featureDim; f++) {
                    sum += inputData.get(b, s, f);
                }
                float mean = sum / featureDim;
                
                // 计算方差
                float variance = 0.0f;
                for (int f = 0; f < featureDim; f++) {
                    float diff = inputData.get(b, s, f) - mean;
                    variance += diff * diff;
                }
                variance = variance / featureDim;
                
                // 计算标准差
                float std = (float) Math.sqrt(variance + epsilon);
                
                // 应用归一化和可学习参数
                for (int f = 0; f < featureDim; f++) {
                    float normalized = (inputData.get(b, s, f) - mean) / std;
                    float result = gamma.getValue().get(0, f) * normalized + beta.getValue().get(0, f);
                    output.set(result, b, s, f);
                }
            }
        }
        
        return new Variable(output);
    }
    
    @Override
    public NdArray forward(NdArray... inputs) {
        return layerForward(new Variable(inputs[0])).getValue();
    }
    
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 层归一化的反向传播比较复杂，这里提供简化版本
        // 在实际应用中，需要计算对输入、gamma和beta的梯度
        List<NdArray> result = new ArrayList<>();
        result.add(yGrad);
        return result;
    }
    
    @Override
    public int requireInputNum() {
        return 1;
    }
    
    /**
     * 获取gamma参数
     */
    public Parameter getGamma() {
        return gamma;
    }
    
    /**
     * 获取beta参数
     */
    public Parameter getBeta() {
        return beta;
    }
}