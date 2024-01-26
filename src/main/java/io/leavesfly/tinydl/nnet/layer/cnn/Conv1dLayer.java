package io.leavesfly.tinydl.nnet.layer.cnn;

import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Layer;
import io.leavesfly.tinydl.nnet.Parameter;

import java.util.Arrays;
import java.util.List;

/**
 * 一维卷积
 */
public class Conv1dLayer extends Layer {
    private Parameter wParam;

    public Conv1dLayer(String _name, int kernelCols, Shape _xInputShape, Shape _yOutputShape) {
        super(_name, _xInputShape, _yOutputShape);

        NdArray initWeight = NdArray.likeRandomN(
                new Shape(1, kernelCols)).mulNumber(Math.sqrt((float) 1 * kernelCols));
        wParam = new Parameter(initWeight);
        wParam.setName("conv1d");
        addParam(wParam.getName(), wParam);
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray input = inputs[0];
        NdArray kernel = inputs[1];

        int inputRows = input.getShape().getRow();
        int inputCols = input.getShape().getColumn();

        int kernelRows = 1;
        int kernelCols = kernel.getShape().getColumn();

        int outputCols = inputCols - kernelCols + 1;
        NdArray output = NdArray.zeros(new Shape(inputRows, outputCols));

        for (int i = 0; i < inputRows; i++) {
            for (int j = 0; j < outputCols; j++) {
                NdArray subInput = input.subNdArray(i, i + kernelRows, j, j + kernelCols);
                float result = subInput.mul(kernel).sum().getNumber().floatValue();
                output.getMatrix()[i][j] = result;
            }
        }
        return output;
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray input = inputs[0].getValue();
        NdArray kernel = inputs[1].getValue();

        NdArray outputNdArray = output.getValue();
        int inputRows = input.getShape().getRow();
        int inputCols = input.getShape().getColumn();
        int kernelRows = 1;
        int kernelCols = kernel.getShape().getColumn();
        int outputRows = outputNdArray.getShape().getRow();
        int outputCols = outputNdArray.getShape().getColumn();

        NdArray inputGradient = NdArray.zeros(new Shape(inputRows, inputCols));

        for (int i = 0; i < outputRows; i++) {
            for (int j = 0; j < outputCols; j++) {
                NdArray subInput = input.subNdArray(i, i + kernelRows, j, j + kernelCols);
                NdArray kernelGradient = subInput.mulNumber(yGrad.getMatrix()[i][j]);
                inputGradient.addTo(i, j, kernelGradient);
            }
        }
        return Arrays.asList(null, inputGradient);
    }

    @Override
    public int requireInputNum() {
        return 2;
    }

    @Override
    public void init() {

    }

    @Override
    public Variable forward(Variable... inputs) {
        return this.call(inputs[0], wParam);
    }
}
