package io.leavesfly.tinydl.func;

import io.leavesfly.tinydl.utils.Config;
import io.leavesfly.tinydl.ndarr.NdArray;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * 抽象的数学函数
 */
public abstract class Function {

    /**
     * 函数的输入变量
     */
    protected Variable[] inputs;

    /**
     * 函数的输出变量
     */
    protected Variable output;

    /**
     * 函数的执行函数，函数执行过程中，
     * 构建了Variable与Function的计算图
     * 记录一个变量是有什么函数以及什么变量运行生成的
     *
     * @param _inputs
     * @return
     */
    public Variable call(Variable... _inputs) {

        if (_inputs.length != requireInputNum() && requireInputNum() > 0) {
            throw new RuntimeException("Function call inputs Variable requireInputNum error!");
        }

        List<NdArray> ndArrayList = Arrays.stream(_inputs).map(Variable::getValue).collect(Collectors.toList());
        NdArray[] ndArrayInputs = ndArrayList.toArray(new NdArray[_inputs.length]);

        //执行函数
        NdArray ndArrayOutput = forward(ndArrayInputs);
        Variable _output = new Variable(ndArrayOutput);

        /**
         * 优化，只有需要向后传播的才会构建计算图
         * 如果没有构建计算图，plot也不能绘制
         */
        if (Config.train) {
            this.inputs = _inputs;
            output = _output;
            output.setCreator(this);
        }
        return _output;
    }

    /**
     * 函数的前向传播
     *
     * @param inputs
     * @return
     */
    public abstract NdArray forward(NdArray... inputs);

    /**
     * 函数的后向传播，求导
     *
     * @param yGrad
     * @return
     */
    public abstract List<NdArray> backward(NdArray yGrad);

    public Variable[] getInputs() {
        return inputs;
    }

    public void setInputs(Variable[] inputs) {
        this.inputs = inputs;
    }


    public Variable getOutput() {
        return output;
    }


    /*
    每个函数一般需要约束输入参数的个数
     */
    public abstract int requireInputNum();
}
