package io.leavesfly.tinydl.nnet.layer.rnn;

import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Parameter;
import io.leavesfly.tinydl.nnet.RnnLayer;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * 递归网络层
 */
public class SimpleRnnLayer extends RnnLayer {

    Parameter x2h;

    Parameter b;

    Parameter h2h;

    private Variable state;
    private NdArray stateValue;

    // 用于反向传播的缓存变量
    private Variable prevState;
    private Variable preTanh;
    private Variable xLinear;
    private Variable hLinear;

    private int hiddeSize;

    public SimpleRnnLayer(String _name, Shape _xInputShape, Shape _yOutputShape) {
        super(_name, _xInputShape, _yOutputShape);
        hiddeSize = _yOutputShape.getColumn();
        init();
    }

    @Override
    public void resetState() {
        state = null;
        stateValue = null;
    }

    @Override
    public void init() {

        int inputSize = inputShape.getColumn();

        NdArray initWeight = NdArray.likeRandomN(new Shape(inputSize, hiddeSize))
                .mulNum(Math.sqrt((double) 1 / inputSize));
        x2h = new Parameter(initWeight);
        x2h.setName("x2h");
        addParam(x2h.getName(), x2h);

        b = new Parameter(NdArray.zeros(new Shape(1, hiddeSize)));
        b.setName("x2h-b");
        addParam(b.getName(), b);


        initWeight = NdArray.likeRandomN(new Shape(hiddeSize, hiddeSize))
                .mulNum(Math.sqrt((double) 1 / hiddeSize));
        h2h = new Parameter(initWeight);
        h2h.setName("h2h");
        addParam(h2h.getName(), h2h);

    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable x = inputs[0];
        if (Objects.isNull(state)) {
            prevState = null;
            xLinear = x.linear(x2h, b);
            state = xLinear.tanh();
            stateValue = state.getValue();
            preTanh = state;
        } else {
            prevState = state;
            xLinear = x.linear(x2h, b);
            hLinear = new Variable(stateValue).linear(h2h, null);
            state = xLinear.add(hLinear).tanh();
            stateValue = state.getValue();
            preTanh = state;
        }
        return state;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray x = inputs[0];
        if (stateValue == null) {
            // 第一次前向传播
            NdArray linearResult = x.dot(x2h.getValue()).add(b.getValue().broadcastTo(x.getShape()));
            stateValue = linearResult.tanh();
        } else {
            // 后续前向传播，包含前一状态
            NdArray xLinear = x.dot(x2h.getValue()).add(b.getValue().broadcastTo(x.getShape()));
            NdArray hLinear = stateValue.dot(h2h.getValue());
            NdArray linearResult = xLinear.add(hLinear);
            stateValue = linearResult.tanh();
        }
        return stateValue;
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 计算tanh的梯度
        NdArray tanhGrad = yGrad.mul(NdArray.ones(preTanh.getValue().getShape()).sub(preTanh.getValue().square()));
        
        // 计算线性变换的梯度
        NdArray xLinearGrad = tanhGrad;
        NdArray hLinearGrad = tanhGrad;
        
        // 计算输入x的梯度
        NdArray xGrad = xLinearGrad.dot(x2h.getValue().transpose());
        
        // 计算参数梯度
        NdArray x2hGrad = inputs[0].getValue().transpose().dot(xLinearGrad);
        NdArray bGrad = xLinearGrad.sumTo(b.getValue().getShape());
        
        NdArray h2hGrad = null;
        NdArray hGrad = null;
        
        if (prevState != null) {
            // 如果有前一状态，计算h2h的梯度
            h2hGrad = prevState.getValue().transpose().dot(hLinearGrad);
            // 计算前一状态的梯度
            hGrad = hLinearGrad.dot(h2h.getValue().transpose());
            // 将梯度相加
            xGrad = xGrad.add(hGrad);
        }
        
        if (prevState != null) {
            return Arrays.asList(xGrad, x2hGrad, bGrad, h2hGrad);
        } else {
            return Arrays.asList(xGrad, x2hGrad, bGrad);
        }
    }

}
