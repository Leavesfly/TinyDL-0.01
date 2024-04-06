package io.leavesfly.tinydl.nnet.layer.cnn;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Layer;
import io.leavesfly.tinydl.nnet.Parameter;

import java.util.Arrays;
import java.util.List;

public class ConvLayer extends Layer {

    private Parameter filterParam;

    //@param stride  步长。
    //@param pad     填充。
    private int stride = 1;
    private int pad = 0;

    private int filterNum;
    private int filterHeight;
    private int filterWidth;
    private int outHeight;
    private int outWidth;

    private NdArray colInput;

    private NdArray colInputW;

    public ConvLayer(String _name, Shape inputShape, int _filterNum, int _filterHeight, int _filterWidth, int _stride, int _pad) {

        super(_name, inputShape);

        if (inputShape.dimension.length != 4) {
            throw new RuntimeException("ConvLayer inputShape error!");
        }

        filterNum = _filterNum;
        filterHeight = _filterHeight;
        filterWidth = _filterWidth;
        stride = _stride;
        pad = _pad;

        int num = inputShape.dimension[0];
//        int channel = inputShape.dimension[1];
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
            //初始化wParam
            int channel = inputShape.dimension[1];
            Shape wParamShape = new Shape(filterNum, channel, filterHeight, filterWidth);

            filterParam = new Parameter(NdArray.likeRandomN(wParamShape));
            filterParam.setName("filterParam");
            addParam(filterParam.getName(), filterParam);

            alreadyInit = true;
        }
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];
        return this.call(input, filterParam);
    }

    @Override
    public NdArray forward(NdArray... inputs) {

        //实现前向传播

        NdArray input = inputs[0];
        int num = input.shape.dimension[0];

        float[][][][] data = input.get4dArray();
        float[][] colInput2dArray = Im2ColUtil.im2col(data, filterHeight, filterWidth, stride, pad);
        colInput = new NdArray(colInput2dArray);

        NdArray filterNdArray = filterParam.getValue();
        colInputW = filterNdArray.reshape(new Shape(filterNum, filterNdArray.shape.size() / filterNum));

        NdArray out = colInput.dot(colInputW);
        out = out.reshape(new Shape(num, outHeight, outWidth, out.shape.size() / (num * outHeight * outWidth)));
        out = out.transpose(0, 3, 1, 2);

        return out;
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {

        //实现后向传播
        int size = yGrad.shape.size();
        NdArray yGradNdArray = yGrad.transpose(0, 2, 3, 1).reshape(new Shape(size / filterNum, filterNum));


        NdArray filterParamGrad = colInput.dot(yGradNdArray);
        filterParamGrad = filterParamGrad.transpose(1, 0).reshape(
                new Shape(filterNum, inputShape.dimension[1], filterHeight, filterWidth));

        NdArray inputXGrad = yGradNdArray.dot(colInputW);

        inputXGrad = new NdArray(
                Col2ImUtil.col2Im(inputXGrad.getMatrix(), inputShape.dimension, filterHeight, filterWidth, stride, pad));
        return Arrays.asList(inputXGrad, filterParamGrad);
    }

}
