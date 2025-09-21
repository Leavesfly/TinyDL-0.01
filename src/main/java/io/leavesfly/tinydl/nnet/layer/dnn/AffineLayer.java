package io.leavesfly.tinydl.nnet.layer.dnn;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Layer;
import io.leavesfly.tinydl.nnet.Parameter;


/**
 * 全连接层与LinearLayer基本一样
 */
public class AffineLayer extends Layer {
    private Parameter wParam;
    private Parameter bParam;
    private boolean needBias;

    public AffineLayer(String _name, Shape _inputShape, int hiddenCol, boolean _needBias) {
        super(_name, _inputShape, new Shape(_inputShape.getRow(), hiddenCol));
        needBias = _needBias;
        //初始化
        init();
    }

    @Override
    public void init() {

        if (!alreadyInit) {
            NdArray initWeight = NdArray.likeRandomN(
                            new Shape(inputShape.getColumn(), outputShape.getColumn()))
                    .mulNum(Math.sqrt((double) 1 / inputShape.getColumn()));

            wParam = new Parameter(initWeight);
            wParam.setName("w");
            addParam(wParam.getName(), wParam);

            if (needBias) {
                bParam = new Parameter(NdArray.zeros(new Shape(1, outputShape.getColumn())));
                bParam.setName("b");
                addParam(bParam.getName(), bParam);
            }
            alreadyInit = true;
        }
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable x = inputs[0];
        return x.linear(wParam, bParam);
    }

}
