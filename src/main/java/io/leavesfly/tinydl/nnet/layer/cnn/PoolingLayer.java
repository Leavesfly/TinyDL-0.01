package io.leavesfly.tinydl.nnet.layer.cnn;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Layer;
import io.leavesfly.tinydl.utils.Util;

import java.util.Collections;
import java.util.List;

/**
 * 池化层
 * 支持最大池化、平均池化和自适应池化
 */
public class PoolingLayer extends Layer {

    public enum PoolingMode {
        MAX, AVERAGE, ADAPTIVE_MAX, ADAPTIVE_AVERAGE
    }

    //@param stride  步长。
    //@param pad     填充。
    private int stride = 1;
    private int pad = 0;

    private int poolHeight;
    private int poolWidth;
    private int outHeight;
    private int outWidth;
    private PoolingMode mode;  // 池化模式

    // 缓存变量
    private NdArray colInput;
    private NdArray argMax;  // 用于最大池化反向传播
    private NdArray input;


    /**
     * 构造函数（默认最大池化）
     */
    public PoolingLayer(String _name, Shape inputShape, int _poolHeight, int _poolWidth, int _stride, int _pad) {
        this(_name, inputShape, _poolHeight, _poolWidth, _stride, _pad, PoolingMode.MAX);
    }

    /**
     * 完整构造函数
     */
    public PoolingLayer(String _name, Shape inputShape, int _poolHeight, int _poolWidth, int _stride, int _pad, PoolingMode _mode) {

        super(_name, inputShape);

        if (inputShape.dimension.length != 4) {
            throw new RuntimeException("PoolingLayer inputShape error!");
        }

        poolHeight = _poolHeight;
        poolWidth = _poolWidth;
        stride = _stride;
        pad = _pad;
        mode = _mode;

        int num = inputShape.dimension[0];
        int channel = inputShape.dimension[1];
        int inHeight = inputShape.dimension[2];
        int inWidth = inputShape.dimension[3];

        if (mode == PoolingMode.ADAPTIVE_MAX || mode == PoolingMode.ADAPTIVE_AVERAGE) {
            // 自适应池化：输出尺寸为 poolHeight x poolWidth
            outHeight = poolHeight;
            outWidth = poolWidth;
        } else {
            // 传统池化
            outHeight = (inHeight + 2 * pad - poolHeight) / stride + 1;
            outWidth = (inWidth + 2 * pad - poolWidth) / stride + 1;
        }
        
        outputShape = new Shape(num, channel, outHeight, outWidth);

        init();
    }

    @Override
    public void init() {

        if (!alreadyInit) {
            alreadyInit = true;
        }
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];
        return this.call(input);
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        // 实现前向传播
        input = inputs[0];
        int num = input.shape.dimension[0];
        int channel = input.shape.dimension[1];
        int inHeight = input.shape.dimension[2];
        int inWidth = input.shape.dimension[3];

        switch (mode) {
            case MAX:
                return forwardMaxPooling(num, channel, inHeight, inWidth);
            case AVERAGE:
                return forwardAveragePooling(num, channel, inHeight, inWidth);
            case ADAPTIVE_MAX:
                return forwardAdaptiveMaxPooling(num, channel, inHeight, inWidth);
            case ADAPTIVE_AVERAGE:
                return forwardAdaptiveAveragePooling(num, channel, inHeight, inWidth);
            default:
                throw new IllegalArgumentException("Unsupported pooling mode: " + mode);
        }
    }
    
    /**
     * 最大池化前向传播
     */
    private NdArray forwardMaxPooling(int num, int channel, int inHeight, int inWidth) {
        float[][][][] data = input.get4dArray();
        float[][] colInput2dArray = Im2ColUtil.im2col(data, poolHeight, poolWidth, stride, pad);
        colInput = new NdArray(colInput2dArray);

        colInput = colInput.reshape(new Shape(colInput.shape.size() / (poolHeight * poolWidth), (poolHeight * poolWidth)));

        argMax = colInput.argMax(1);
        NdArray out = colInput.max(1);
        out = out.reshape(new Shape(num, outHeight, outWidth, channel));
        
        // 手动转置维度从 (N, H, W, C) 到 (N, C, H, W)
        return transposeNHWCToNCHW(out, num, channel, outHeight, outWidth);
    }
    
    /**
     * 平均池化前向传播
     */
    private NdArray forwardAveragePooling(int num, int channel, int inHeight, int inWidth) {
        float[][][][] data = input.get4dArray();
        float[][] colInput2dArray = Im2ColUtil.im2col(data, poolHeight, poolWidth, stride, pad);
        colInput = new NdArray(colInput2dArray);

        colInput = colInput.reshape(new Shape(colInput.shape.size() / (poolHeight * poolWidth), (poolHeight * poolWidth)));

        // 计算平均值
        NdArray out = computeMean(colInput);
        out = out.reshape(new Shape(num, outHeight, outWidth, channel));
        
        return transposeNHWCToNCHW(out, num, channel, outHeight, outWidth);
    }
    
    /**
     * 自适应最大池化前向传播
     */
    private NdArray forwardAdaptiveMaxPooling(int num, int channel, int inHeight, int inWidth) {
        NdArray result = new NdArray(new Shape(num, channel, outHeight, outWidth));
        argMax = new NdArray(new Shape(num, channel, outHeight, outWidth));
        
        for (int n = 0; n < num; n++) {
            for (int c = 0; c < channel; c++) {
                for (int oh = 0; oh < outHeight; oh++) {
                    for (int ow = 0; ow < outWidth; ow++) {
                        int hStart = (int) Math.floor((double) oh * inHeight / outHeight);
                        int hEnd = (int) Math.ceil((double) (oh + 1) * inHeight / outHeight);
                        int wStart = (int) Math.floor((double) ow * inWidth / outWidth);
                        int wEnd = (int) Math.ceil((double) (ow + 1) * inWidth / outWidth);
                        
                        float maxVal = Float.NEGATIVE_INFINITY;
                        int maxIdx = 0;
                        
                        for (int h = hStart; h < hEnd; h++) {
                            for (int w = wStart; w < wEnd; w++) {
                                int idx = ((n * channel + c) * inHeight + h) * inWidth + w;
                                if (input.buffer[idx] > maxVal) {
                                    maxVal = input.buffer[idx];
                                    maxIdx = idx;
                                }
                            }
                        }
                        
                        int resultIdx = ((n * channel + c) * outHeight + oh) * outWidth + ow;
                        result.buffer[resultIdx] = maxVal;
                        argMax.buffer[resultIdx] = maxIdx;
                    }
                }
            }
        }
        
        return result;
    }
    
    /**
     * 自适应平均池化前向传播
     */
    private NdArray forwardAdaptiveAveragePooling(int num, int channel, int inHeight, int inWidth) {
        NdArray result = new NdArray(new Shape(num, channel, outHeight, outWidth));
        
        for (int n = 0; n < num; n++) {
            for (int c = 0; c < channel; c++) {
                for (int oh = 0; oh < outHeight; oh++) {
                    for (int ow = 0; ow < outWidth; ow++) {
                        int hStart = (int) Math.floor((double) oh * inHeight / outHeight);
                        int hEnd = (int) Math.ceil((double) (oh + 1) * inHeight / outHeight);
                        int wStart = (int) Math.floor((double) ow * inWidth / outWidth);
                        int wEnd = (int) Math.ceil((double) (ow + 1) * inWidth / outWidth);
                        
                        float sum = 0.0f;
                        int count = 0;
                        
                        for (int h = hStart; h < hEnd; h++) {
                            for (int w = wStart; w < wEnd; w++) {
                                int idx = ((n * channel + c) * inHeight + h) * inWidth + w;
                                sum += input.buffer[idx];
                                count++;
                            }
                        }
                        
                        int resultIdx = ((n * channel + c) * outHeight + oh) * outWidth + ow;
                        result.buffer[resultIdx] = sum / count;
                    }
                }
            }
        }
        
        return result;
    }
    
    /**
     * 计算平均值
     */
    private NdArray computeMean(NdArray input) {
        int rows = input.shape.dimension[0];
        int cols = input.shape.dimension[1];
        NdArray result = new NdArray(new Shape(rows));
        
        for (int i = 0; i < rows; i++) {
            float sum = 0.0f;
            for (int j = 0; j < cols; j++) {
                sum += input.buffer[i * cols + j];
            }
            result.buffer[i] = sum / cols;
        }
        
        return result;
    }
    
    /**
     * 维度转置：(N, H, W, C) -> (N, C, H, W)
     */
    private NdArray transposeNHWCToNCHW(NdArray input, int num, int channel, int height, int width) {
        NdArray result = new NdArray(new Shape(num, channel, height, width));
        for (int n = 0; n < num; n++) {
            for (int c = 0; c < channel; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        float value = input.get(n, h, w, c);
                        result.set(value, n, c, h, w);
                    }
                }
            }
        }
        return result;
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 实现后向传播
        switch (mode) {
            case MAX:
                return backwardMaxPooling(yGrad);
            case AVERAGE:
                return backwardAveragePooling(yGrad);
            case ADAPTIVE_MAX:
                return backwardAdaptiveMaxPooling(yGrad);
            case ADAPTIVE_AVERAGE:
                return backwardAdaptiveAveragePooling(yGrad);
            default:
                throw new IllegalArgumentException("Unsupported pooling mode: " + mode);
        }
    }
    
    /**
     * 最大池化反向传播
     */
    private List<NdArray> backwardMaxPooling(NdArray yGrad) {
        yGrad = yGrad.transpose(0, 2, 3, 1);

        int poolSize = poolHeight * poolWidth;
        int size = yGrad.shape.size();
        NdArray dMax = NdArray.zeros(new Shape(size, poolSize));

        // 优化的梯度分发
        NdArray flatten = yGrad.flatten();
        int[] colSlices = Util.toInt(flatten.buffer);

        dMax.setItem(Util.getSeq(argMax.shape.size()), colSlices, flatten.buffer);

        dMax = dMax.reshape(new Shape(yGrad.shape.dimension[0], yGrad.shape.dimension[1], yGrad.shape.dimension[2]
                , yGrad.shape.dimension[3], poolSize));

        int dMaxSize = dMax.shape.dimension[0] * dMax.shape.dimension[1] * dMax.shape.dimension[2];
        NdArray dCol = dMax.reshape(new Shape(dMaxSize, dMax.shape.size() / dMaxSize));

        float[][][][] data = Col2ImUtil.col2im(dCol.getMatrix(), input.shape.dimension, poolHeight, poolWidth, stride, pad);
        NdArray inputXGrad = new NdArray(data);
        return Collections.singletonList(inputXGrad);
    }
    
    /**
     * 平均池化反向传播
     */
    private List<NdArray> backwardAveragePooling(NdArray yGrad) {
        yGrad = yGrad.transpose(0, 2, 3, 1);

        int poolSize = poolHeight * poolWidth;
        int size = yGrad.shape.size();
        
        // 平均池化的梯度平均分配给每个元素
        NdArray dAvg = new NdArray(new Shape(size, poolSize));
        NdArray flattenGrad = yGrad.flatten();
        
        for (int i = 0; i < size; i++) {
            float avgGrad = flattenGrad.buffer[i] / poolSize;
            for (int j = 0; j < poolSize; j++) {
                dAvg.buffer[i * poolSize + j] = avgGrad;
            }
        }

        dAvg = dAvg.reshape(new Shape(yGrad.shape.dimension[0], yGrad.shape.dimension[1], yGrad.shape.dimension[2]
                , yGrad.shape.dimension[3], poolSize));

        int dAvgSize = dAvg.shape.dimension[0] * dAvg.shape.dimension[1] * dAvg.shape.dimension[2];
        NdArray dCol = dAvg.reshape(new Shape(dAvgSize, dAvg.shape.size() / dAvgSize));

        float[][][][] data = Col2ImUtil.col2im(dCol.getMatrix(), input.shape.dimension, poolHeight, poolWidth, stride, pad);
        NdArray inputXGrad = new NdArray(data);
        return Collections.singletonList(inputXGrad);
    }
    
    /**
     * 自适应最大池化反向传播
     */
    private List<NdArray> backwardAdaptiveMaxPooling(NdArray yGrad) {
        int num = input.shape.dimension[0];
        int channel = input.shape.dimension[1];
        int inHeight = input.shape.dimension[2];
        int inWidth = input.shape.dimension[3];
        
        NdArray inputGrad = NdArray.zeros(input.shape);
        
        for (int n = 0; n < num; n++) {
            for (int c = 0; c < channel; c++) {
                for (int oh = 0; oh < outHeight; oh++) {
                    for (int ow = 0; ow < outWidth; ow++) {
                        int gradIdx = ((n * channel + c) * outHeight + oh) * outWidth + ow;
                        int maxIdx = (int) argMax.buffer[gradIdx];
                        inputGrad.buffer[maxIdx] += yGrad.buffer[gradIdx];
                    }
                }
            }
        }
        
        return Collections.singletonList(inputGrad);
    }
    
    /**
     * 自适应平均池化反向传播
     */
    private List<NdArray> backwardAdaptiveAveragePooling(NdArray yGrad) {
        int num = input.shape.dimension[0];
        int channel = input.shape.dimension[1];
        int inHeight = input.shape.dimension[2];
        int inWidth = input.shape.dimension[3];
        
        NdArray inputGrad = NdArray.zeros(input.shape);
        
        for (int n = 0; n < num; n++) {
            for (int c = 0; c < channel; c++) {
                for (int oh = 0; oh < outHeight; oh++) {
                    for (int ow = 0; ow < outWidth; ow++) {
                        int hStart = (int) Math.floor((double) oh * inHeight / outHeight);
                        int hEnd = (int) Math.ceil((double) (oh + 1) * inHeight / outHeight);
                        int wStart = (int) Math.floor((double) ow * inWidth / outWidth);
                        int wEnd = (int) Math.ceil((double) (ow + 1) * inWidth / outWidth);
                        
                        int gradIdx = ((n * channel + c) * outHeight + oh) * outWidth + ow;
                        float gradValue = yGrad.buffer[gradIdx];
                        int poolSize = (hEnd - hStart) * (wEnd - wStart);
                        float avgGrad = gradValue / poolSize;
                        
                        for (int h = hStart; h < hEnd; h++) {
                            for (int w = wStart; w < wEnd; w++) {
                                int inputIdx = ((n * channel + c) * inHeight + h) * inWidth + w;
                                inputGrad.buffer[inputIdx] += avgGrad;
                            }
                        }
                    }
                }
            }
        }
        
        return Collections.singletonList(inputGrad);
    }
}
