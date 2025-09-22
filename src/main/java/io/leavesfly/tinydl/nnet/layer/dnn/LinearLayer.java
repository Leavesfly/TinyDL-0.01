package io.leavesfly.tinydl.nnet.layer.dnn;

import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Layer;
import io.leavesfly.tinydl.nnet.Parameter;

import java.util.List;

/**
 * 线性层（全连接层）
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * LinearLayer实现了一个标准的全连接层，对输入进行线性变换。
 * 该层执行以下计算：y = x * W + b
 * 其中W是权重矩阵，b是偏置项（可选）。
 */
public class LinearLayer extends Layer {
    /**
     * 权重参数矩阵
     * 形状: (input_size, output_size)
     */
    private Parameter w;
    
    /**
     * 偏置参数向量
     * 形状: (1, output_size)
     */
    private Parameter b;

    /**
     * 构造一个线性层实例
     * 
     * @param _name 层名称
     * @param hiddenRow 输入维度（行数）
     * @param hiddenCol 输出维度（列数）
     * @param needBias 是否需要偏置项
     */
    public LinearLayer(String _name, int hiddenRow, int hiddenCol, boolean needBias) {
        super(_name, new Shape(-1, hiddenRow), new Shape(-1, hiddenCol));
        NdArray initWeight = NdArray.likeRandomN(new Shape(hiddenRow, hiddenCol)).mulNum(Math.sqrt((double) 1 / hiddenRow));
        w = new Parameter(initWeight);
        w.setName("w");
        addParam(w.getName(), w);

        if (needBias) {
            b = new Parameter(NdArray.zeros(new Shape(1, hiddenCol)));
            b.setName("b");
            addParam(b.getName(), b);
        }
    }

    /**
     * 初始化方法（空实现，参数已在构造函数中初始化）
     */
    @Override
    public void init() {

    }

    /**
     * 线性层的前向传播方法
     * 
     * @param inputs 输入变量数组，通常只包含一个输入变量
     * @return 线性变换后的输出变量
     */
    @Override
    public Variable layerForward(Variable... inputs) {
        return inputs[0].linear(w, b);
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
        return 0;
    }
}