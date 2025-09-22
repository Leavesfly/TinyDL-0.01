package io.leavesfly.tinydl.nnet.layer.cnn;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Layer;
import io.leavesfly.tinydl.nnet.Parameter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 卷积层实现类
 * 
 * 实现了标准的卷积操作，支持步长、填充、偏置等参数。
 * 使用Im2Col技术将卷积操作转换为矩阵乘法，提高计算效率。
 */
public class ConvLayer extends Layer {

    // 卷积核参数
    private Parameter filterParam;
    
    // 偏置参数
    private Parameter biasParam;

    // 卷积操作参数
    private int stride;
    private int padding;

    // 卷积核尺寸
    private int filterNum;
    private int filterHeight;
    private int filterWidth;
    
    // 输出特征图尺寸
    private int outHeight;
    private int outWidth;

    // 是否使用偏置
    private boolean useBias;

    // 缓存变量，提高内存效率
    private NdArray colInput;
    private NdArray colInputWeight;
    
    /**
     * 构造函数（不使用偏置）
     * 
     * @param name 层名称
     * @param inputShape 输入形状 [batchSize, channels, height, width]
     * @param filterNum 卷积核数量
     * @param filterHeight 卷积核高度
     * @param filterWidth 卷积核宽度
     * @param stride 步长
     * @param padding 填充
     */
    public ConvLayer(String name, Shape inputShape, int filterNum, int filterHeight, int filterWidth, int stride, int padding) {
        this(name, inputShape, filterNum, filterHeight, filterWidth, stride, padding, false);
    }

    /**
     * 完整构造函数
     * 
     * @param name 层名称
     * @param inputShape 输入形状 [batchSize, channels, height, width]
     * @param filterNum 卷积核数量
     * @param filterHeight 卷积核高度
     * @param filterWidth 卷积核宽度
     * @param stride 步长
     * @param padding 填充
     * @param useBias 是否使用偏置
     */
    public ConvLayer(String name, Shape inputShape, int filterNum, int filterHeight, int filterWidth, int stride, int padding, boolean useBias) {
        super(name, inputShape);

        // 验证输入形状
        validateInputShape(inputShape);

        // 初始化参数
        this.filterNum = filterNum;
        this.filterHeight = filterHeight;
        this.filterWidth = filterWidth;
        this.stride = stride;
        this.padding = padding;
        this.useBias = useBias;

        // 计算输出尺寸
        calculateOutputDimensions(inputShape);

        // 设置输出形状
        int batchSize = inputShape.dimension[0];
        this.outputShape = new Shape(batchSize, filterNum, outHeight, outWidth);

        // 初始化参数
        init();
    }

    /**
     * 验证输入形状是否符合卷积层要求
     * 
     * @param inputShape 输入形状
     * @throws RuntimeException 当输入形状不正确时抛出异常
     */
    private void validateInputShape(Shape inputShape) {
        if (inputShape.dimension.length != 4) {
            throw new RuntimeException("ConvLayer inputShape must be 4D: [batchSize, channels, height, width], but got " + 
                Arrays.toString(inputShape.dimension));
        }
    }

    /**
     * 计算输出特征图的尺寸
     * 
     * @param inputShape 输入形状
     */
    private void calculateOutputDimensions(Shape inputShape) {
        int inputHeight = inputShape.dimension[2];
        int inputWidth = inputShape.dimension[3];
        
        this.outHeight = (inputHeight + 2 * padding - filterHeight) / stride + 1;
        this.outWidth = (inputWidth + 2 * padding - filterWidth) / stride + 1;
    }

    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化权重参数（使用Xavier初始化）
            initializeWeights();
            
            // 初始化偏置参数（如果使用）
            if (useBias) {
                initializeBias();
            }

            alreadyInit = true;
        }
    }

    /**
     * 初始化权重参数（使用Xavier初始化）
     */
    private void initializeWeights() {
        int inputChannels = inputShape.dimension[1];
        
        // Xavier初始化：标准差 = sqrt(2 / (fan_in + fan_out))
        int fanIn = inputChannels * filterHeight * filterWidth;
        int fanOut = filterNum * filterHeight * filterWidth;
        float stddev = (float) Math.sqrt(2.0f / (fanIn + fanOut));
        
        // 创建权重参数形状
        Shape weightShape = new Shape(filterNum, inputChannels, filterHeight, filterWidth);
        
        // 使用正态分布初始化，然后缩放到合适的标准差
        filterParam = new Parameter(NdArray.likeRandomN(weightShape));
        
        // 缩放到Xavier标准差
        NdArray weights = filterParam.getValue();
        for (int i = 0; i < weights.buffer.length; i++) {
            weights.buffer[i] *= stddev;
        }
        
        filterParam.setName("filterParam");
        addParam(filterParam.getName(), filterParam);
    }

    /**
     * 初始化偏置参数
     */
    private void initializeBias() {
        Shape biasShape = new Shape(filterNum);
        biasParam = new Parameter(NdArray.zeros(biasShape));
        biasParam.setName("biasParam");
        addParam(biasParam.getName(), biasParam);
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];
        if (useBias && biasParam != null) {
            return this.call(input, filterParam, biasParam);
        } else {
            return this.call(input, filterParam);
        }
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        // 获取输入数据
        NdArray input = inputs[0];
        int batchSize = input.shape.dimension[0];

        // 执行Im2Col操作，将输入转换为列格式
        colInput = performIm2Col(input);
        
        // 重塑权重参数以进行矩阵乘法
        colInputWeight = reshapeFilterWeights();

        // 执行矩阵乘法
        NdArray output = performMatrixMultiplication();
        
        // 重塑输出为正确的形状
        NdArray result = reshapeOutput(output, batchSize);
        
        // 添加偏置（如果使用）
        if (useBias) {
            NdArray bias = (inputs.length > 1) ? inputs[1] : biasParam.getValue();
            result = addBiasToOutput(result, bias);
        }
        
        return result;
    }
    
    /**
     * 执行Im2Col操作
     * 
     * @param input 输入数据
     * @return 列格式的输入数据
     */
    private NdArray performIm2Col(NdArray input) {
        float[][][][] inputData = input.get4dArray();
        float[][] colInput2dArray = Im2ColUtil.im2col(inputData, filterHeight, filterWidth, stride, padding);
        return new NdArray(colInput2dArray);
    }
    
    /**
     * 重塑滤波器权重以进行矩阵乘法
     * 
     * @return 重塑后的权重
     */
    private NdArray reshapeFilterWeights() {
        NdArray filterNdArray = filterParam.getValue();
        return filterNdArray.reshape(new Shape(filterNum, filterNdArray.shape.size() / filterNum));
    }
    
    /**
     * 执行矩阵乘法操作
     * 
     * @return 矩阵乘法结果
     */
    private NdArray performMatrixMultiplication() {
        return colInput.dot(colInputWeight.transpose());
    }
    
    /**
     * 重塑输出为正确的形状
     * 
     * @param output 矩阵乘法的输出
     * @param batchSize 批处理大小
     * @return 重塑后的输出
     */
    private NdArray reshapeOutput(NdArray output, int batchSize) {
        NdArray result = new NdArray(new Shape(batchSize, filterNum, outHeight, outWidth));
        int resultIndex = 0;
        
        for (int batch = 0; batch < batchSize; batch++) {
            for (int channel = 0; channel < filterNum; channel++) {
                for (int height = 0; height < outHeight; height++) {
                    for (int width = 0; width < outWidth; width++) {
                        int outputIndex = (batch * outHeight + height) * outWidth + width;
                        result.buffer[resultIndex++] = output.buffer[outputIndex * filterNum + channel];
                    }
                }
            }
        }
        
        return result;
    }
    
    /**
     * 向输出添加偏置
     * 
     * @param output 输出数据
     * @param bias 偏置参数
     * @return 添加偏置后的输出
     */
    private NdArray addBiasToOutput(NdArray output, NdArray bias) {
        int batchSize = output.shape.dimension[0];
        int channels = output.shape.dimension[1];
        int height = output.shape.dimension[2];
        int width = output.shape.dimension[3];
        
        NdArray result = new NdArray(output.buffer.clone(), output.shape);
        
        for (int batch = 0; batch < batchSize; batch++) {
            for (int channel = 0; channel < channels; channel++) {
                float biasValue = bias.buffer[channel];
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int index = ((batch * channels + channel) * height + h) * width + w;
                        result.buffer[index] += biasValue;
                    }
                }
            }
        }
        
        return result;
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 计算权重梯度
        NdArray weightGrad = computeWeightGradient(yGrad);
        
        // 计算输入梯度
        NdArray inputGrad = computeInputGradient(yGrad);
        
        // 构建梯度列表
        List<NdArray> gradients = new ArrayList<>();
        gradients.add(inputGrad);
        gradients.add(weightGrad);
        
        // 计算偏置梯度（如果使用）
        if (useBias) {
            NdArray biasGrad = computeBiasGradient(yGrad);
            gradients.add(biasGrad);
        }
        
        return gradients;
    }
    
    /**
     * 计算权重梯度
     * 
     * @param yGrad 输出梯度
     * @return 权重梯度
     */
    private NdArray computeWeightGradient(NdArray yGrad) {
        int size = yGrad.shape.size();
        NdArray yGradReshaped = yGrad.transpose(0, 2, 3, 1).reshape(new Shape(size / filterNum, filterNum));
        NdArray weightGrad = colInput.dot(yGradReshaped);
        return weightGrad.transpose(1, 0).reshape(new Shape(filterNum, inputShape.dimension[1], filterHeight, filterWidth));
    }
    
    /**
     * 计算输入梯度
     * 
     * @param yGrad 输出梯度
     * @return 输入梯度
     */
    private NdArray computeInputGradient(NdArray yGrad) {
        int size = yGrad.shape.size();
        NdArray yGradReshaped = yGrad.transpose(0, 2, 3, 1).reshape(new Shape(size / filterNum, filterNum));
        NdArray inputGrad = yGradReshaped.dot(colInputWeight);
        float[][][][] data = Col2ImUtil.col2im(inputGrad.getMatrix(), inputShape.dimension, filterHeight, filterWidth, stride, padding);
        return new NdArray(data);
    }
    
    /**
     * 计算偏置梯度
     * 
     * @param yGrad 输出梯度
     * @return 偏置梯度
     */
    private NdArray computeBiasGradient(NdArray yGrad) {
        int batchSize = yGrad.shape.dimension[0];
        int channels = yGrad.shape.dimension[1];
        int height = yGrad.shape.dimension[2];
        int width = yGrad.shape.dimension[3];
        
        NdArray biasGrad = NdArray.zeros(new Shape(channels));
        
        for (int channel = 0; channel < channels; channel++) {
            float sum = 0.0f;
            for (int batch = 0; batch < batchSize; batch++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int index = ((batch * channels + channel) * height + h) * width + w;
                        sum += yGrad.buffer[index];
                    }
                }
            }
            biasGrad.buffer[channel] = sum;
        }
        
        return biasGrad;
    }
}