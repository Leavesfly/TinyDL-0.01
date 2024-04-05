package io.leavesfly.tinydl.nnet.layer.cnn;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Layer;

import java.util.List;

//todo
public class PoolingLayer extends Layer {
    private int winWidth, winHeight;
    private int strideX, strideY;
    private int[][] maxIdx;


    public PoolingLayer(String _name, int winSize, int stride, Shape _xInputShape, Shape _yOutputShape) {
        super(_name, _xInputShape, _yOutputShape);
        this.winWidth = winSize;
        this.winHeight = winSize;
        this.strideX = stride;
        this.strideY = stride;
    }

    @Override
    public NdArray forward(NdArray... inputs) {

        return null;
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {

        return null;
    }

    @Override
    public int requireInputNum() {
        return 1;
    }

    @Override
    public void init() {

    }

    @Override
    public Variable forward(Variable... inputs) {
        return this.call(inputs[0]);
    }
}
