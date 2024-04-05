package io.leavesfly.tinydl.nnet.layer.cnn;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Layer;
import io.leavesfly.tinydl.nnet.Parameter;

import java.util.List;

//todo 待完善
public class ConvLayer extends Layer {

    private Parameter wParameter;

    private int winWidth, winHeight;
    private int strideX, strideY;
    private int paddingX, paddingY;
    private int filterCount;


    public ConvLayer(String _name, int winWidth, int winHeight, int strideX, int strideY, int filterCount, PaddingType type, Shape _xInputShape, Shape _yOutputShape) {

        super(_name, _xInputShape, _yOutputShape);

        if (type == PaddingType.VALID) {
            this.winWidth = winWidth;
            this.winHeight = winHeight;
            this.strideX = strideX;
            this.strideY = strideY;
            this.filterCount = filterCount;
            this.paddingX = 0;
            this.paddingY = 0;

        } else {
            this.winWidth = winWidth;
            this.winHeight = winHeight;
            this.strideX = strideX;
            this.strideY = strideY;
            this.filterCount = filterCount;

            if ((winWidth - 1) % 2 != 0) {
                throw new RuntimeException("Bad sizes for convolution!");
            }
            this.paddingX = (winWidth - 1) / 2;

            if ((winHeight - 1) % 2 != 0) {
                throw new RuntimeException("Bad sizes for convolution!");
            }
            this.paddingY = (winHeight - 1) / 2;
        }

        init();
    }

    @Override
    public void init() {

    }

    public ConvLayer(String name, int winSize, int stride, int filterCount, PaddingType type, Shape _xInputShape, Shape _yOutputShape) {
        this(name, winSize, winSize, stride, stride, filterCount, type, _xInputShape, _yOutputShape);
    }

    public ConvLayer(String name, int winSize, int filterCount, PaddingType type, Shape _xInputShape, Shape _yOutputShape) {
        this(name, winSize, 1, filterCount, type, _xInputShape, _yOutputShape);
    }


    @Override
    public NdArray forward(NdArray... inputs) {

        //实现前向传播

        return null;
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {

        //实现后向传播

        return null;
    }

    @Override
    public int requireInputNum() {
        return 0;
    }


    public enum PaddingType {
        VALID, SAME;
    }
}
