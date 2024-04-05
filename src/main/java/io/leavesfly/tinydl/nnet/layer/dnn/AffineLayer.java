package io.leavesfly.tinydl.nnet.layer.dnn;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Layer;
import io.leavesfly.tinydl.nnet.Parameter;

import java.util.List;

/**
 * 全连接层与LinearLayer基本一样
 */
public class AffineLayer extends Layer {
    private Parameter wParam;
    private Parameter bParam;
    private boolean needBias;

    public AffineLayer(String _name, Shape inputXShape, int hiddenCol, boolean _needBias) {
        super(_name, inputXShape, new Shape(inputXShape.getColumn(), hiddenCol));
        needBias = _needBias;

        init();
    }

    @Override
    public void init() {

        if (!alreadyInit) {
            NdArray initWeight = NdArray.likeRandomN(
                            new Shape(xInputShape.getRow(), yOutputShape.getColumn()))
                    .mulNum(Math.sqrt((double) 1 / xInputShape.getRow()));
            wParam = new Parameter(initWeight);
            wParam.setName("w");
            addParam(wParam.getName(), wParam);

            if (needBias) {
                bParam = new Parameter(NdArray.zeros(new Shape(1, yOutputShape.getColumn())));
                bParam.setName("b");
                addParam(bParam.getName(), bParam);
            }
            alreadyInit = true;
        }
    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];
        return x.linear(wParam, bParam);
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        return null;
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return null;
    }

}
