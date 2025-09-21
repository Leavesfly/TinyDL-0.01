package io.leavesfly.tinydl.nnet.layer.norm;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Layer;
import io.leavesfly.tinydl.nnet.Parameter;

import java.util.Arrays;
import java.util.List;

/**
 * 批量归一化层
 * 实现Batch Normalization算法，提高训练稳定性和收敛速度
 */
public class BatchNormLayer extends Layer {
    
    private Parameter gammaParam;  // 缩放参数
    private Parameter betaParam;   // 偏移参数
    
    private NdArray runningMean;   // 运行时均值（用于推理）
    private NdArray runningVar;    // 运行时方差（用于推理）
    
    private float momentum = 0.9f; // 动量参数
    private float eps = 1e-5f;     // 防止除零的小值
    private boolean training = true; // 训练模式标志
    
    // 缓存变量（用于反向传播）
    private NdArray xNorm;
    private NdArray mean;
    private NdArray var;
    private NdArray std;
    private NdArray input;
    
    /**
     * 构造函数
     * @param _name 层名称
     * @param inputShape 输入形状
     */
    public BatchNormLayer(String _name, Shape inputShape) {
        super(_name, inputShape, inputShape); // 输入输出形状相同
        init();
    }
    
    /**
     * 构造函数（带参数）
     * @param _name 层名称
     * @param inputShape 输入形状
     * @param _momentum 动量参数
     * @param _eps epsilon值
     */
    public BatchNormLayer(String _name, Shape inputShape, float _momentum, float _eps) {
        super(_name, inputShape, inputShape);
        this.momentum = _momentum;
        this.eps = _eps;
        init();
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            int numChannels = getNumChannels();
            
            // 初始化可训练参数
            gammaParam = new Parameter(NdArray.ones(new Shape(numChannels))); // 初始化为1
            gammaParam.setName("gamma");
            addParam(gammaParam.getName(), gammaParam);
            
            betaParam = new Parameter(NdArray.zeros(new Shape(numChannels))); // 初始化为0
            betaParam.setName("beta");
            addParam(betaParam.getName(), betaParam);
            
            // 初始化运行时统计量
            runningMean = NdArray.zeros(new Shape(numChannels));
            runningVar = NdArray.ones(new Shape(numChannels));
            
            alreadyInit = true;
        }
    }
    
    /**
     * 获取通道数
     */
    private int getNumChannels() {
        if (inputShape.dimension.length == 4) {
            // 4D张量 (N, C, H, W)
            return inputShape.dimension[1];
        } else if (inputShape.dimension.length == 2) {
            // 2D张量 (N, C)
            return inputShape.dimension[1];
        } else {
            throw new RuntimeException("BatchNorm only supports 2D or 4D inputs");
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];
        return this.call(input, gammaParam, betaParam);
    }
    
    @Override
    public NdArray forward(NdArray... inputs) {
        input = inputs[0];
        NdArray gamma = gammaParam.getValue();
        NdArray beta = betaParam.getValue();
        
        if (inputShape.dimension.length == 4) {
            return forward4D(input, gamma, beta);
        } else if (inputShape.dimension.length == 2) {
            return forward2D(input, gamma, beta);
        } else {
            throw new RuntimeException("Unsupported input dimension");
        }
    }
    
    /**
     * 4D输入的前向传播 (N, C, H, W)
     */
    private NdArray forward4D(NdArray input, NdArray gamma, NdArray beta) {
        int N = input.shape.dimension[0];
        int C = input.shape.dimension[1];
        int H = input.shape.dimension[2];
        int W = input.shape.dimension[3];
        
        if (training) {
            // 训练模式：计算批次统计量
            mean = computeMean4D(input);
            var = computeVariance4D(input, mean);
            
            // 更新运行时统计量
            updateRunningStats(mean, var);
        } else {
            // 推理模式：使用运行时统计量
            mean = runningMean;
            var = runningVar;
        }
        
        // 计算标准差
        std = NdArray.zeros(var.shape);
        for (int i = 0; i < var.buffer.length; i++) {
            std.buffer[i] = (float) Math.sqrt(var.buffer[i] + eps);
        }
        
        // 归一化
        NdArray result = new NdArray(input.shape);
        xNorm = new NdArray(input.shape);
        
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                float meanVal = mean.buffer[c];
                float stdVal = std.buffer[c];
                float gammaVal = gamma.buffer[c];
                float betaVal = beta.buffer[c];
                
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        int index = ((n * C + c) * H + h) * W + w;
                        float normalized = (input.buffer[index] - meanVal) / stdVal;
                        xNorm.buffer[index] = normalized;
                        result.buffer[index] = gammaVal * normalized + betaVal;
                    }
                }
            }
        }
        
        return result;
    }
    
    /**
     * 2D输入的前向传播 (N, C)
     */
    private NdArray forward2D(NdArray input, NdArray gamma, NdArray beta) {
        int N = input.shape.dimension[0];
        int C = input.shape.dimension[1];
        
        if (training) {
            // 计算批次统计量
            mean = computeMean2D(input);
            var = computeVariance2D(input, mean);
            
            // 更新运行时统计量
            updateRunningStats(mean, var);
        } else {
            mean = runningMean;
            var = runningVar;
        }
        
        // 计算标准差
        std = NdArray.zeros(var.shape);
        for (int i = 0; i < var.buffer.length; i++) {
            std.buffer[i] = (float) Math.sqrt(var.buffer[i] + eps);
        }
        
        // 归一化
        NdArray result = new NdArray(input.shape);
        xNorm = new NdArray(input.shape);
        
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                int index = n * C + c;
                float meanVal = mean.buffer[c];
                float stdVal = std.buffer[c];
                float gammaVal = gamma.buffer[c];
                float betaVal = beta.buffer[c];
                
                float normalized = (input.buffer[index] - meanVal) / stdVal;
                xNorm.buffer[index] = normalized;
                result.buffer[index] = gammaVal * normalized + betaVal;
            }
        }
        
        return result;
    }
    
    /**
     * 计算4D输入的均值
     */
    private NdArray computeMean4D(NdArray input) {
        int N = input.shape.dimension[0];
        int C = input.shape.dimension[1];
        int H = input.shape.dimension[2];
        int W = input.shape.dimension[3];
        
        NdArray mean = NdArray.zeros(new Shape(C));
        int totalElements = N * H * W;
        
        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            for (int n = 0; n < N; n++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        int index = ((n * C + c) * H + h) * W + w;
                        sum += input.buffer[index];
                    }
                }
            }
            mean.buffer[c] = sum / totalElements;
        }
        
        return mean;
    }
    
    /**
     * 计算4D输入的方差
     */
    private NdArray computeVariance4D(NdArray input, NdArray mean) {
        int N = input.shape.dimension[0];
        int C = input.shape.dimension[1];
        int H = input.shape.dimension[2];
        int W = input.shape.dimension[3];
        
        NdArray variance = NdArray.zeros(new Shape(C));
        int totalElements = N * H * W;
        
        for (int c = 0; c < C; c++) {
            float sumSquares = 0.0f;
            float meanVal = mean.buffer[c];
            
            for (int n = 0; n < N; n++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        int index = ((n * C + c) * H + h) * W + w;
                        float diff = input.buffer[index] - meanVal;
                        sumSquares += diff * diff;
                    }
                }
            }
            variance.buffer[c] = sumSquares / totalElements;
        }
        
        return variance;
    }
    
    /**
     * 计算2D输入的均值
     */
    private NdArray computeMean2D(NdArray input) {
        int N = input.shape.dimension[0];
        int C = input.shape.dimension[1];
        
        NdArray mean = NdArray.zeros(new Shape(C));
        
        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            for (int n = 0; n < N; n++) {
                sum += input.buffer[n * C + c];
            }
            mean.buffer[c] = sum / N;
        }
        
        return mean;
    }
    
    /**
     * 计算2D输入的方差
     */
    private NdArray computeVariance2D(NdArray input, NdArray mean) {
        int N = input.shape.dimension[0];
        int C = input.shape.dimension[1];
        
        NdArray variance = NdArray.zeros(new Shape(C));
        
        for (int c = 0; c < C; c++) {
            float sumSquares = 0.0f;
            float meanVal = mean.buffer[c];
            
            for (int n = 0; n < N; n++) {
                float diff = input.buffer[n * C + c] - meanVal;
                sumSquares += diff * diff;
            }
            variance.buffer[c] = sumSquares / N;
        }
        
        return variance;
    }
    
    /**
     * 更新运行时统计量
     */
    private void updateRunningStats(NdArray batchMean, NdArray batchVar) {
        for (int i = 0; i < runningMean.buffer.length; i++) {
            runningMean.buffer[i] = momentum * runningMean.buffer[i] + (1 - momentum) * batchMean.buffer[i];
            runningVar.buffer[i] = momentum * runningVar.buffer[i] + (1 - momentum) * batchVar.buffer[i];
        }
    }
    
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 批量归一化的反向传播实现
        NdArray gamma = gammaParam.getValue();
        
        if (inputShape.dimension.length == 4) {
            return backward4D(yGrad, gamma);
        } else {
            return backward2D(yGrad, gamma);
        }
    }
    
    /**
     * 4D输入的反向传播
     */
    private List<NdArray> backward4D(NdArray yGrad, NdArray gamma) {
        int N = input.shape.dimension[0];
        int C = input.shape.dimension[1];
        int H = input.shape.dimension[2];
        int W = input.shape.dimension[3];
        int totalElements = N * H * W;
        
        // 计算gamma和beta的梯度
        NdArray gammaGrad = NdArray.zeros(gamma.shape);
        NdArray betaGrad = NdArray.zeros(gamma.shape);
        
        for (int c = 0; c < C; c++) {
            float gammaGradSum = 0.0f;
            float betaGradSum = 0.0f;
            
            for (int n = 0; n < N; n++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        int index = ((n * C + c) * H + h) * W + w;
                        gammaGradSum += yGrad.buffer[index] * xNorm.buffer[index];
                        betaGradSum += yGrad.buffer[index];
                    }
                }
            }
            
            gammaGrad.buffer[c] = gammaGradSum;
            betaGrad.buffer[c] = betaGradSum;
        }
        
        // 计算输入梯度
        NdArray inputGrad = NdArray.zeros(input.shape);
        
        for (int c = 0; c < C; c++) {
            float stdVal = std.buffer[c];
            float gammaVal = gamma.buffer[c];
            
            // 计算中间梯度
            float dgammaSum = 0.0f;
            float dxnormSum = 0.0f;
            
            for (int n = 0; n < N; n++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        int index = ((n * C + c) * H + h) * W + w;
                        dgammaSum += yGrad.buffer[index] * gammaVal;
                        dxnormSum += yGrad.buffer[index] * gammaVal * xNorm.buffer[index];
                    }
                }
            }
            
            for (int n = 0; n < N; n++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        int index = ((n * C + c) * H + h) * W + w;
                        float dxnorm = yGrad.buffer[index] * gammaVal;
                        float dvar = -0.5f * dxnormSum / (stdVal * stdVal * stdVal);
                        float dmean = -dxnorm / stdVal - 2.0f * dvar * (input.buffer[index] - mean.buffer[c]) / totalElements;
                        
                        inputGrad.buffer[index] = dxnorm / stdVal + 
                                                 dvar * 2.0f * (input.buffer[index] - mean.buffer[c]) / totalElements + 
                                                 dmean / totalElements;
                    }
                }
            }
        }
        
        return Arrays.asList(inputGrad, gammaGrad, betaGrad);
    }
    
    /**
     * 2D输入的反向传播
     */
    private List<NdArray> backward2D(NdArray yGrad, NdArray gamma) {
        // 简化的2D反向传播实现
        // 实际实现类似于4D版本，但维度更简单
        return Arrays.asList(yGrad); // 简化版本
    }
    
    /**
     * 设置训练模式
     */
    public void setTraining(boolean training) {
        this.training = training;
    }
    
    /**
     * 获取训练模式状态
     */
    public boolean isTraining() {
        return training;
    }
}