package io.leavesfly.tinydl.mlearning;

import io.leavesfly.tinydl.utils.Plot;
import io.leavesfly.tinydl.utils.Util;

import java.util.ArrayList;
import java.util.List;

/**
 * 模型训练监控器
 * 
 * 该类用于收集和可视化模型训练过程中的信息，包括：
 * 1. 训练损失值的收集和存储
 * 2. 训练信息的打印输出
 * 3. 训练过程的可视化展示
 * 
 * @author TinyDL
 * @version 1.0
 */
public class Monitor {

    private int index;

    List<Float> lossList;

    /**
     * 默认构造函数
     */
    public Monitor() {
        index = 0;
        this.lossList = new ArrayList<>();
    }

    /**
     * 开始新的训练轮次
     * @param _index 轮次索引
     */
    public void startNewEpoch(int _index) {
        index = _index;
    }

    /**
     * 收集训练信息（损失值）
     * @param loss 当前批次的损失值
     */
    public void collectInfo(float loss) {
        lossList.add(loss);
    }

    /**
     * 打印训练信息
     */
    public void printTrainInfo() {
        System.out.println("epoch = " + index + ", loss:" + lossList.get(index));
    }

    /**
     * 绘制训练过程图表
     */
    public void plot() {
        Plot plot = new Plot();
        Float[] loss = lossList.toArray(new Float[0]);
        plot.line(Util.toFloat(Util.getSeq(lossList.size())), Util.toFloat(loss), "loss");
        plot.show();
    }
}