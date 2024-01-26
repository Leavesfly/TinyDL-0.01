package io.leavesfly.tinydl.func;

import io.leavesfly.tinydl.func.base.*;
import io.leavesfly.tinydl.func.loss.MeanSE;
import io.leavesfly.tinydl.func.loss.SoftmaxCE;

import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.func.math.*;
import io.leavesfly.tinydl.func.matrix.*;

import java.util.List;
import java.util.Objects;

/**
 * 数学中的变量的抽象
 */
public class Variable {
    /**
     * 变量的名称
     */
    private String name;

    /**
     * 变量的值
     */
    private NdArray value;

    /**
     * 变量的梯度
     */
    private NdArray grad;

    /**
     * 记录是什么函数生成的当前Variable
     */
    private Function creator;

    /**
     * 是否需要计算当前变量的梯度
     */
    private boolean requireGrad = true;

    /**
     * 构造函数
     *
     * @param _value
     */
    public Variable(NdArray _value) {
        if (Objects.isNull(_value)) {
            throw new RuntimeException("NdArray value is null!");
        }
        this.value = _value;
    }

    public Variable(Number number) {
        if (Objects.isNull(number)) {
            throw new RuntimeException("NdArray number is null!");
        }
        this.value = new NdArray(number);
    }

    public Variable(NdArray _value, String _name) {
        if (Objects.isNull(_value)) {
            throw new RuntimeException("NdArray _value is null!");
        }
        this.value = _value;
        this.name = _name;
    }

    public Variable(NdArray _value, String _name, boolean _requireGrad) {
        if (Objects.isNull(_value)) {
            throw new RuntimeException("NdArray _value is null!");
        }
        this.value = _value;
        this.name = _name;
        this.requireGrad = _requireGrad;
    }

    public Variable setRequireGrad(boolean _requireGrad) {
        this.requireGrad = _requireGrad;
        return this;
    }

    /**
     * 变量的反向传播，会根据正向传播时构建的计算图，
     * 反向传播计算每个变量的梯度
     */
    public void backward() {
        if (!requireGrad) {
            this.grad = null;
            return;
        }
        //初始化为1
        if (Objects.isNull(grad)) {
            setGrad(NdArray.ones(this.getValue().getShape()));
        }
        // todo 当前采用的是递归调用，为了效率可用堆栈循环
        Function _creator = creator;
        if (!Objects.isNull(_creator)) {
            Variable[] _inputs = _creator.getInputs();
            List<NdArray> grads = _creator.backward(grad);
            if (_inputs.length != grads.size()) {
                throw new RuntimeException("Variable backward grads size error!");
            }
            int index = 0;
            for (Variable input : _inputs) {
                input.setGrad(grads.get(index));
                input.backward();
                index++;
            }
        }
    }


    /**
     * 用在RNN中，切断计算图
     */
    public void unChainBackward() {
        Function creatorFunc = creator;
        if (!Objects.isNull(creatorFunc)) {
            Variable[] xs = creatorFunc.getInputs();
            unChain();
            for (Variable x : xs) {
                x.unChainBackward();
            }
        }
    }

    /**
     * 清理梯度
     */
    public void clearGrad() {
        grad = null;
    }

    public NdArray getValue() {
        return value;
    }

    private void unChain() {
        creator = null;
    }

    public void setValue(NdArray value) {
        this.value = value;
    }


    public NdArray getGrad() {
        return grad;
    }

    public void setGrad(NdArray _grad) {
        if (_grad == null) {
            return;
        }
        if (!_grad.getShape().equals(value.getShape())) {
            throw new RuntimeException("_grad shape must equal value shape!");
        }
        if (requireGrad) {
            this.grad = _grad;
        } else {
            this.grad = null;
        }
    }

    public Function getCreator() {
        return creator;
    }

    public void setCreator(Function creator) {
        this.creator = creator;
    }

    public String getName() {
        return name;
    }

    public Variable setName(String name) {
        this.name = name;
        return this;
    }


    //    # =============================================================================
//            # 1，四则运算
//    # =============================================================================

    public Variable add(Variable other) {
        Function function = new Add();
        return function.call(this, other);
    }

    public Variable sub(Variable other) {
        Function function = new Sub();
        return function.call(this, other);
    }

    public Variable mul(Variable other) {
        Function function = new Mul();
        return function.call(this, other);
    }

    public Variable div(Variable other) {
        Function function = new Div();
        return function.call(this, other);
    }

    /**
     * 取反
     */
    public Variable neg() {
        Function function = new Neg();
        return function.call(this);
    }


    //    # =============================================================================
//            # 2，基本数学函数
//    # =============================================================================

    public Variable squ() {
        Function function = new Squ();
        return function.call(this);
    }

    public Variable pow(float pow) {
        Function function = new Pow(pow);
        return function.call(this);
    }

    public Variable exp() {
        Function function = new Exp();
        return function.call(this);
    }

    public Variable sin() {
        Function function = new Sin();
        return function.call(this);
    }

    public Variable cos() {
        Function function = new Cos();
        return function.call(this);
    }

    public Variable log() {
        Function function = new Log();
        return function.call(this);
    }

    public Variable tanh() {
        Function function = new Tanh();
        return function.call(this);
    }

    public Variable softMax() {
        Function function = new SoftMax();
        return function.call(this);
    }

    public Variable clip(float min, float max) {
        Function function = new Clip(min, max);
        return function.call(this);
    }

    public Variable max(int _axis, boolean _keepdims) {
        Function function = new Max(_axis, _keepdims);
        return function.call(this);
    }

    public Variable min(int _axis, boolean _keepdims) {
        Function function = new Min(_axis, _keepdims);
        return function.call(this);
    }


    //    # =============================================================================
//            # 3，张量的变形操作
//    # =============================================================================


    public Variable broadcastTo(Shape shape) {
        Function function = new BroadcastTo(shape);
        return function.call(this);
    }

    public Variable matMul(Variable other) {
        Function function = new MatMul();
        return function.call(this, other);
    }

    public Variable reshape(Shape shape) {
        Function function = new Reshape(shape);
        return function.call(this);
    }

    public Variable sum() {
        Function function = new Sum();
        return function.call(this);
    }

    public Variable sumTo(Shape shape) {
        Function function = new SumTo(shape);
        return function.call(this);
    }

    public Variable transpose() {
        Function function = new Transpose();
        return function.call(this);
    }

    public Variable linear(Variable w, Variable b) {
        Function function = new Linear();
        if (Objects.isNull(b)) {
            return function.call(this, w);
        }
        return function.call(this, w, b);
    }

    public Variable getItem(int[] _rowSlices, int[] _colSlices) {
        Function function = new GetItem(_rowSlices, _colSlices);
        return function.call(this);
    }

    //    # =============================================================================
//            # 4，loss函数
//    # =============================================================================
    public Variable meanSquaredError(Variable other) {
        Function function = new MeanSE();
        return function.call(this, other);
    }

    public Variable softmaxCrossEntropy(Variable other) {
        return new SoftmaxCE().call(this, other);
    }
}
