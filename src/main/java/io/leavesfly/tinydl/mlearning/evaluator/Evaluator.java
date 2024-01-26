package io.leavesfly.tinydl.mlearning.evaluator;

import io.leavesfly.tinydl.mlearning.Model;
import io.leavesfly.tinydl.mlearning.dataset.DataSet;

/**
 * 效果评估器
 */
public abstract class Evaluator {

    protected Model model;
    protected DataSet dataSet;

    public abstract void evaluate();

}
