package io.leavesfly.tinydl.modality.nlp.layer;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Layer;
import io.leavesfly.tinydl.nnet.layer.activate.ReLuLayer;
import io.leavesfly.tinydl.nnet.layer.dnn.LinearLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * MoE专家网络实现
 * 
 * 每个专家是一个独立的前馈神经网络，具有自己的参数。
 * 专家网络的结构与标准的FeedForward层相同：
 * Expert(x) = ReLU(xW1 + b1)W2 + b2
 * 
 * 在MoE中，每个专家专注于处理特定类型的输入模式，
 * 通过门控网络的路由机制来选择最合适的专家。
 * 
 * @author leavesfly
 * @version 0.01
 */
public class MoEExpertNetwork extends Layer {
    
    /**
     * 第一个线性层：输入到隐藏层
     */
    private LinearLayer firstLinear;
    
    /**
     * 激活函数层
     */
    private ReLuLayer activation;
    
    /**
     * 第二个线性层：隐藏层到输出
     */
    private LinearLayer secondLinear;
    
    /**
     * 输入维度
     */
    private int inputDim;
    
    /**
     * 隐藏层维度
     */
    private int hiddenDim;
    
    /**
     * 输出维度
     */
    private int outputDim;
    
    /**
     * 专家ID（用于标识和调试）
     */
    private int expertId;
    
    /**
     * Dropout比率（可选）
     */
    private double dropoutRate;
    
    /**
     * 构造专家网络
     * 
     * @param name 专家名称
     * @param expertId 专家ID
     * @param inputDim 输入维度
     * @param hiddenDim 隐藏层维度
     * @param outputDim 输出维度
     * @param dropoutRate Dropout比率
     */
    public MoEExpertNetwork(String name, int expertId, int inputDim, int hiddenDim, int outputDim, double dropoutRate) {
        super(name, new Shape(-1, -1, inputDim), new Shape(-1, -1, outputDim));
        this.expertId = expertId;
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.outputDim = outputDim;
        this.dropoutRate = dropoutRate;
        init();
    }
    
    /**
     * 简化构造函数，使用默认的隐藏层维度（4倍输入维度）
     * 
     * @param name 专家名称
     * @param expertId 专家ID
     * @param modelDim 模型维度（输入和输出维度相同）
     */
    public MoEExpertNetwork(String name, int expertId, int modelDim) {
        this(name, expertId, modelDim, modelDim * 4, modelDim, 0.0);
    }
    
    /**
     * 带Dropout的简化构造函数
     * 
     * @param name 专家名称
     * @param expertId 专家ID
     * @param modelDim 模型维度
     * @param dropoutRate Dropout比率
     */
    public MoEExpertNetwork(String name, int expertId, int modelDim, double dropoutRate) {
        this(name, expertId, modelDim, modelDim * 4, modelDim, dropoutRate);
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 第一个线性层：inputDim -> hiddenDim
            firstLinear = new LinearLayer(
                name + "_expert" + expertId + "_linear1", 
                inputDim, 
                hiddenDim, 
                true
            );
            
            // ReLU激活函数
            activation = new ReLuLayer(
                name + "_expert" + expertId + "_relu", 
                new Shape(-1, -1, hiddenDim)
            );
            
            // 第二个线性层：hiddenDim -> outputDim
            secondLinear = new LinearLayer(
                name + "_expert" + expertId + "_linear2", 
                hiddenDim, 
                outputDim, 
                true
            );
            
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
        
        // 重塑为二维以进行矩阵运算
        NdArray inputReshaped = inputData.reshape(new Shape(batchSize * seqLen, inputDim));
        Variable reshapedInput = new Variable(inputReshaped);
        
        // 第一个线性变换
        Variable hidden = firstLinear.layerForward(reshapedInput);
        
        // ReLU激活
        Variable activated = activation.layerForward(hidden);
        
        // 应用Dropout（训练时）
        if (dropoutRate > 0.0) {
            activated = applyDropout(activated);
        }
        
        // 第二个线性变换
        Variable output2D = secondLinear.layerForward(activated);
        
        // 重塑回三维
        NdArray output3D = output2D.getValue().reshape(new Shape(batchSize, seqLen, outputDim));
        
        return new Variable(output3D);
    }
    
    /**
     * 应用Dropout正则化
     * 简化版本：在训练时随机将一些神经元置零
     * 
     * @param input 输入变量
     * @return 应用Dropout后的变量
     */
    private Variable applyDropout(Variable input) {
        if (dropoutRate <= 0.0) {
            return input;
        }
        
        NdArray inputData = input.getValue();
        NdArray droppedData = NdArray.zeros(inputData.shape);
        
        // 简化的Dropout实现
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
     * 计算专家网络的参数数量
     * 
     * @return 参数数量
     */
    public long getParameterCount() {
        long params = 0;
        
        // 第一个线性层参数：(inputDim * hiddenDim) + hiddenDim
        params += (long) inputDim * hiddenDim + hiddenDim;
        
        // 第二个线性层参数：(hiddenDim * outputDim) + outputDim
        params += (long) hiddenDim * outputDim + outputDim;
        
        return params;
    }
    
    /**
     * 克隆专家网络（用于创建具有相同结构但不同参数的专家）
     * 
     * @param newName 新专家的名称
     * @param newExpertId 新专家的ID
     * @return 新的专家网络实例
     */
    public MoEExpertNetwork clone(String newName, int newExpertId) {
        return new MoEExpertNetwork(newName, newExpertId, inputDim, hiddenDim, outputDim, dropoutRate);
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
     * 获取专家ID
     */
    public int getExpertId() {
        return expertId;
    }
    
    /**
     * 获取第一个线性层
     */
    public LinearLayer getFirstLinear() {
        return firstLinear;
    }
    
    /**
     * 获取激活函数层
     */
    public ReLuLayer getActivation() {
        return activation;
    }
    
    /**
     * 获取第二个线性层
     */
    public LinearLayer getSecondLinear() {
        return secondLinear;
    }
    
    /**
     * 获取输入维度
     */
    public int getInputDim() {
        return inputDim;
    }
    
    /**
     * 获取隐藏层维度
     */
    public int getHiddenDim() {
        return hiddenDim;
    }
    
    /**
     * 获取输出维度
     */
    public int getOutputDim() {
        return outputDim;
    }
    
    /**
     * 获取Dropout比率
     */
    public double getDropoutRate() {
        return dropoutRate;
    }
    
    /**
     * 设置Dropout比率
     */
    public void setDropoutRate(double dropoutRate) {
        this.dropoutRate = Math.max(0.0, Math.min(1.0, dropoutRate));
    }
}