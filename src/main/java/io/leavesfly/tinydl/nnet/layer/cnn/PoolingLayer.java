package io.leavesfly.tinydl.nnet.layer.cnn;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Layer;
import io.leavesfly.tinydl.nnet.Parameter;

import java.util.Arrays;
import java.util.List;

public class PoolingLayer extends Layer {


    //@param stride  步长。
    //@param pad     填充。
    private int stride = 1;
    private int pad = 0;

    private int poolHeight;
    private int poolWidth;
    private int outHeight;
    private int outWidth;

    private NdArray colInput;

    private NdArray argMax;


    public PoolingLayer(String _name, Shape inputShape, int _poolHeight, int _poolWidth, int _stride, int _pad) {

        super(_name, inputShape);

        if (inputShape.dimension.length != 4) {
            throw new RuntimeException("ConvLayer inputShape error!");
        }

        poolHeight = _poolHeight;
        poolWidth = _poolWidth;
        stride = _stride;
        pad = _pad;

        int num = inputShape.dimension[0];
        int channel = inputShape.dimension[1];
        int inHeight = inputShape.dimension[2];
        int inWidth = inputShape.dimension[3];

        outHeight = (inHeight + 2 * pad - poolHeight) / stride + 1;
        outWidth = (inWidth + 2 * pad - poolWidth) / stride + 1;
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

        //实现前向传播
        NdArray input = inputs[0];
        int num = input.shape.dimension[0];
        int channel = input.shape.dimension[1];

        float[][][][] data = input.get4dArray();
        float[][] colInput2dArray = Im2ColUtil.im2col(data, poolHeight, poolWidth, stride, pad);
        colInput = new NdArray(colInput2dArray);

        colInput = colInput.reshape(new Shape(colInput.shape.size() / (poolHeight * poolWidth), (poolHeight * poolWidth)));

        argMax = colInput.argMax(1);
        NdArray out = colInput.max(1);
        out = out.reshape(new Shape(num, outHeight, outWidth, channel)).transpose(0, 3, 1, 2);
        return out;
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {

        //实现后向传播
        int size = yGrad.shape.size();
        NdArray yGradNdArray = yGrad.transpose(0, 2, 3, 1);
        int poolSize = poolHeight * poolWidth;

        NdArray dMax = NdArray.zeros(new Shape(size, poolSize));

        //todo


        NdArray inputXGrad = null;

        return Arrays.asList(inputXGrad);
    }

}
