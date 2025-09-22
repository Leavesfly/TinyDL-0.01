package io.leavesfly.tinydl.nnet.layer.cnn;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Layer;
import io.leavesfly.tinydl.nnet.Parameter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ConvLayer extends Layer {

    private Parameter filterParam;
    private Parameter biasParam;  // 新增偏置参数

    //@param stride  步长。
    //@param pad     填充。
    private int stride = 1;
    private int pad = 0;

    private int filterNum;
    private int filterHeight;
    private int filterWidth;
    private int outHeight;
    private int outWidth;

    private boolean useBias;  // 是否使用偏置

    // 缓存变量，提高内存效率
    private NdArray colInput;
    private NdArray colInputW;
//    private NdArray reshapedOutput;

    /**
     * 构造函数（不使用偏置）
     * @param _name
     * @param inputShape
     * @param _filterNum
     * @param _filterHeight
     * @param _filterWidth
     * @param _stride
     * @param _pad
     */
    public ConvLayer(String _name, Shape inputShape, int _filterNum, int _filterHeight, int _filterWidth, int _stride, int _pad) {
        this(_name, inputShape, _filterNum, _filterHeight, _filterWidth, _stride, _pad, false);
    }

    /**
     * 完整构造函数
     * @param _name
     * @param inputShape
     * @param _filterNum
     * @param _filterHeight
     * @param _filterWidth
     * @param _stride
     * @param _pad
     * @param _useBias 是否使用偏置
     */
    public ConvLayer(String _name, Shape inputShape, int _filterNum, int _filterHeight, int _filterWidth, int _stride, int _pad, boolean _useBias) {

        super(_name, inputShape);

        if (inputShape.dimension.length != 4) {
            throw new RuntimeException("ConvLayer inputShape error!");
        }

        filterNum = _filterNum;
        filterHeight = _filterHeight;
        filterWidth = _filterWidth;
        stride = _stride;
        pad = _pad;
        useBias = _useBias;

        int num = inputShape.dimension[0];
        int inHeight = inputShape.dimension[2];
        int inWidth = inputShape.dimension[3];

        outHeight = (inHeight + 2 * pad - filterHeight) / stride + 1;
        outWidth = (inWidth + 2 * pad - filterWidth) / stride + 1;

        outputShape = new Shape(num, filterNum, outHeight, outWidth);

        init();
    }

    @Override
    public void init() {

        if (!alreadyInit) {
            //初始化权重参数（使用Xavier初始化）
            int channel = inputShape.dimension[1];
            Shape wParamShape = new Shape(filterNum, channel, filterHeight, filterWidth);
            
            // Xavier初始化：标准差 = sqrt(2 / (fan_in + fan_out))
            int fanIn = channel * filterHeight * filterWidth;
            int fanOut = filterNum * filterHeight * filterWidth;
            float stddev = (float) Math.sqrt(2.0f / (fanIn + fanOut));
            
            // 使用正态分布初始化，然后缩放到合适的标准差
            filterParam = new Parameter(NdArray.likeRandomN(wParamShape));
            // 缩放到Xavier标准差
            NdArray weights = filterParam.getValue();
            for (int i = 0; i < weights.buffer.length; i++) {
                weights.buffer[i] *= stddev;
            }
            filterParam.setName("filterParam");
            addParam(filterParam.getName(), filterParam);

            // 初始化偏置参数（如果使用）
            if (useBias) {
                Shape biasShape = new Shape(filterNum);
                biasParam = new Parameter(NdArray.zeros(biasShape));
                biasParam.setName("biasParam");
                addParam(biasParam.getName(), biasParam);
            }

            alreadyInit = true;
        }
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];
        if (useBias) {
            return this.call(input, filterParam, biasParam);
        } else {
            return this.call(input, filterParam);
        }
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        // 实现前向传播
        NdArray input = inputs[0];
        int num = input.shape.dimension[0];

        // Im2Col操作
        float[][][][] data = input.get4dArray();
        float[][] colInput2dArray = Im2ColUtil.im2col(data, filterHeight, filterWidth, stride, pad);
        colInput = new NdArray(colInput2dArray);

        // 权重变形
        NdArray filterNdArray = filterParam.getValue();
        colInputW = filterNdArray.reshape(new Shape(filterNum, filterNdArray.shape.size() / filterNum));

        // 矩阵乘法
        NdArray out = colInput.dot(colInputW.transpose());
        
        // 维度变换：优化版本，直接计算索引
        NdArray result = new NdArray(new Shape(num, filterNum, outHeight, outWidth));
        int resultIndex = 0;
        for (int n = 0; n < num; n++) {
            for (int c = 0; c < filterNum; c++) {
                for (int h = 0; h < outHeight; h++) {
                    for (int w = 0; w < outWidth; w++) {
                        int outIndex = (n * outHeight + h) * outWidth + w;
                        result.buffer[resultIndex++] = out.buffer[outIndex * filterNum + c];
                    }
                }
            }
        }
        
        // 添加偏置（如果使用）
        if (useBias && inputs.length > 1) {
            NdArray bias = inputs[1];
            addBias(result, bias);
        } else if (useBias && biasParam != null) {
            NdArray bias = biasParam.getValue();
            addBias(result, bias);
        }
        
        return result;
    }
    
    /**
     * 添加偏置到输出结果
     * @param result 输出结果
     * @param bias 偏置参数
     */
    private void addBias(NdArray result, NdArray bias) {
        int num = result.shape.dimension[0];
        int channels = result.shape.dimension[1];
        int height = result.shape.dimension[2];
        int width = result.shape.dimension[3];
        
        for (int n = 0; n < num; n++) {
            for (int c = 0; c < channels; c++) {
                float biasValue = bias.buffer[c];
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int index = ((n * channels + c) * height + h) * width + w;
                        result.buffer[index] += biasValue;
                    }
                }
            }
        }
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 实现后向传播
        int size = yGrad.shape.size();
        NdArray yGradNdArray = yGrad.transpose(0, 2, 3, 1).reshape(new Shape(size / filterNum, filterNum));

        // 权重梯度
        NdArray filterParamGrad = colInput.dot(yGradNdArray);
        filterParamGrad = filterParamGrad.transpose(1, 0).reshape(new Shape(filterNum, inputShape.dimension[1], filterHeight, filterWidth));

        // 输入梯度
        NdArray inputXGrad = yGradNdArray.dot(colInputW);
        float[][][][] data = Col2ImUtil.col2im(inputXGrad.getMatrix(), inputShape.dimension, filterHeight, filterWidth, stride, pad);
        inputXGrad = new NdArray(data);

        List<NdArray> gradients = new ArrayList<>();
        gradients.add(inputXGrad);
        gradients.add(filterParamGrad);
        
        // 偏置梯度（如果使用）
        if (useBias) {
            NdArray biasGrad = computeBiasGradient(yGrad);
            gradients.add(biasGrad);
        }
        
        return gradients;
    }
    
    /**
     * 计算偏置梯度
     * @param yGrad 输出梯度
     * @return 偏置梯度
     */
    private NdArray computeBiasGradient(NdArray yGrad) {
        int num = yGrad.shape.dimension[0];
        int channels = yGrad.shape.dimension[1];
        int height = yGrad.shape.dimension[2];
        int width = yGrad.shape.dimension[3];
        
        NdArray biasGrad = NdArray.zeros(new Shape(channels));
        
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            for (int n = 0; n < num; n++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int index = ((n * channels + c) * height + h) * width + w;
                        sum += yGrad.buffer[index];
                    }
                }
            }
            biasGrad.buffer[c] = sum;
        }
        
        return biasGrad;
    }

}
