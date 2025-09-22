package io.leavesfly.tinydl.nnet.layer.rnn;

import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Parameter;
import io.leavesfly.tinydl.nnet.Layer;
import io.leavesfly.tinydl.nnet.layer.activate.SigmoidLayer;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * 门控循环单元层(GRU)
 * GRU是LSTM的简化版本，包含更新门和重置门
 */
public class GruLayer extends Layer {

    private Variable state;
    private NdArray stateValue;

    // GRU的参数
    // 更新门参数
    private Parameter w_z;  // 输入到更新门的权重
    private Parameter u_z;  // 隐藏状态到更新门的权重
    private Parameter b_z;  // 更新门的偏置

    // 重置门参数
    private Parameter w_r;  // 输入到重置门的权重
    private Parameter u_r;  // 隐藏状态到重置门的权重
    private Parameter b_r;  // 重置门的偏置

    // 候选状态参数
    private Parameter w_h;  // 输入到候选状态的权重
    private Parameter u_h;  // 隐藏状态到候选状态的权重
    private Parameter b_h;  // 候选状态的偏置

    private int hiddenSize;

    // 用于反向传播的缓存变量
    private Variable zGate;     // 更新门
    private Variable rGate;     // 重置门
    private Variable hCandidate; // 候选状态
    private Variable resetState; // 重置后的状态

    public GruLayer(String _name, Shape _xInputShape, Shape _yOutputShape) {
        super(_name, _xInputShape, _yOutputShape);
        hiddenSize = _yOutputShape.getColumn();
        init();
    }

    public void resetState() {
        state = null;
        stateValue = null;
    }

    @Override
    public void init() {
        int inputSize = inputShape.getColumn();

        // 初始化更新门参数
        NdArray initWeight = NdArray.likeRandomN(new Shape(inputSize, hiddenSize))
                .mulNum((float) Math.sqrt(2.0 / (inputSize + hiddenSize)));
        w_z = new Parameter(initWeight);
        w_z.setName("w_z");
        addParam(w_z.getName(), w_z);

        initWeight = NdArray.likeRandomN(new Shape(hiddenSize, hiddenSize))
                .mulNum((float) Math.sqrt(2.0 / (hiddenSize + hiddenSize)));
        u_z = new Parameter(initWeight);
        u_z.setName("u_z");
        addParam(u_z.getName(), u_z);

        b_z = new Parameter(NdArray.zeros(new Shape(1, hiddenSize)));
        b_z.setName("b_z");
        addParam(b_z.getName(), b_z);

        // 初始化重置门参数
        initWeight = NdArray.likeRandomN(new Shape(inputSize, hiddenSize))
                .mulNum((float) Math.sqrt(2.0 / (inputSize + hiddenSize)));
        w_r = new Parameter(initWeight);
        w_r.setName("w_r");
        addParam(w_r.getName(), w_r);

        initWeight = NdArray.likeRandomN(new Shape(hiddenSize, hiddenSize))
                .mulNum((float) Math.sqrt(2.0 / (hiddenSize + hiddenSize)));
        u_r = new Parameter(initWeight);
        u_r.setName("u_r");
        addParam(u_r.getName(), u_r);

        b_r = new Parameter(NdArray.zeros(new Shape(1, hiddenSize)));
        b_r.setName("b_r");
        addParam(b_r.getName(), b_r);

        // 初始化候选状态参数
        initWeight = NdArray.likeRandomN(new Shape(inputSize, hiddenSize))
                .mulNum((float) Math.sqrt(2.0 / (inputSize + hiddenSize)));
        w_h = new Parameter(initWeight);
        w_h.setName("w_h");
        addParam(w_h.getName(), w_h);

        initWeight = NdArray.likeRandomN(new Shape(hiddenSize, hiddenSize))
                .mulNum((float) Math.sqrt(2.0 / (hiddenSize + hiddenSize)));
        u_h = new Parameter(initWeight);
        u_h.setName("u_h");
        addParam(u_h.getName(), u_h);

        b_h = new Parameter(NdArray.zeros(new Shape(1, hiddenSize)));
        b_h.setName("b_h");
        addParam(b_h.getName(), b_h);
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable x = inputs[0];

        if (Objects.isNull(state)) {
            // 第一次前向传播
            // 计算更新门
            Variable x_z = x.linear(w_z, b_z);
            zGate = new SigmoidLayer("").call(x_z);

            // 计算重置门
            Variable x_r = x.linear(w_r, b_r);
            rGate = new SigmoidLayer("").call(x_r);

            // 计算候选状态
            Variable x_h = x.linear(w_h, b_h);
            hCandidate = x_h.tanh();

            // 计算当前状态
            Variable oneMinusZ = new Variable(NdArray.ones(zGate.getValue().getShape())).sub(zGate);
            state = oneMinusZ.mul(hCandidate);
            stateValue = state.getValue();
        } else {
            // 后续前向传播
            // 计算更新门
            Variable x_z = x.linear(w_z, b_z);
            Variable h_z = state.linear(u_z, null);
            zGate = new SigmoidLayer("").call(x_z.add(h_z));

            // 计算重置门
            Variable x_r = x.linear(w_r, b_r);
            Variable h_r = state.linear(u_r, null);
            rGate = new SigmoidLayer("").call(x_r.add(h_r));

            // 重置前一状态
            resetState = rGate.mul(state);

            // 计算候选状态
            Variable x_h = x.linear(w_h, b_h);
            Variable h_h = resetState.linear(u_h, null);
            hCandidate = x_h.add(h_h).tanh();

            // 计算当前状态
            Variable oneMinusZ = new Variable(NdArray.ones(zGate.getValue().getShape())).sub(zGate);
            state = zGate.mul(state).add(oneMinusZ.mul(hCandidate));
            stateValue = state.getValue();
        }

        return state;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray x = inputs[0];

        if (stateValue == null) {
            // 第一次前向传播
            // 计算更新门
            NdArray x_z = x.dot(w_z.getValue()).add(b_z.getValue().broadcastTo(x.getShape()));
            NdArray zGateValue = x_z.sigmoid();

            // 计算重置门
            NdArray x_r = x.dot(w_r.getValue()).add(b_r.getValue().broadcastTo(x.getShape()));
            NdArray rGateValue = x_r.sigmoid();

            // 计算候选状态
            NdArray x_h = x.dot(w_h.getValue()).add(b_h.getValue().broadcastTo(x.getShape()));
            NdArray hCandidateValue = x_h.tanh();

            // 计算当前状态
            NdArray oneMinusZ = NdArray.ones(zGateValue.getShape()).sub(zGateValue);
            stateValue = oneMinusZ.mul(hCandidateValue);
        } else {
            // 后续前向传播
            // 计算更新门
            NdArray x_z = x.dot(w_z.getValue()).add(b_z.getValue().broadcastTo(x.getShape()));
            NdArray h_z = stateValue.dot(u_z.getValue());
            NdArray zGateValue = x_z.add(h_z).sigmoid();

            // 计算重置门
            NdArray x_r = x.dot(w_r.getValue()).add(b_r.getValue().broadcastTo(x.getShape()));
            NdArray h_r = stateValue.dot(u_r.getValue());
            NdArray rGateValue = x_r.add(h_r).sigmoid();

            // 重置前一状态
            NdArray resetStateValue = rGateValue.mul(stateValue);

            // 计算候选状态
            NdArray x_h = x.dot(w_h.getValue()).add(b_h.getValue().broadcastTo(x.getShape()));
            NdArray h_h = resetStateValue.dot(u_h.getValue());
            NdArray hCandidateValue = x_h.add(h_h).tanh();

            // 计算当前状态
            NdArray oneMinusZ = NdArray.ones(zGateValue.getShape()).sub(zGateValue);
            stateValue = zGateValue.mul(stateValue).add(oneMinusZ.mul(hCandidateValue));
        }

        return stateValue;
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // TODO: 实现GRU的反向传播
        // 由于GRU的反向传播较为复杂，这里暂时返回null
        // 在实际应用中需要根据GRU的数学公式实现完整的反向传播
        return null;
    }
}