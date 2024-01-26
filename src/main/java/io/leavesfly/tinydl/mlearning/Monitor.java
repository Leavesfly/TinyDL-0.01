package io.leavesfly.tinydl.mlearning;

import io.leavesfly.tinydl.utils.Plot;
import io.leavesfly.tinydl.utils.Util;

import java.util.ArrayList;
import java.util.List;

/**
 * 收集模型训练过程中情况
 */
public class Monitor {

    private int index;

    List<Float> lossList;

    public Monitor() {
        index = 0;
        this.lossList = new ArrayList<>();
    }

    public void startNewEpoch(int _index) {
        index = _index;
    }

    public void collectInfo(float loss) {
        lossList.add(loss);
    }

    public void printTrainInfo() {
        System.out.println("epoch = " + index + ", loss:" + lossList.get(index));
    }

    public void plot() {
        Plot plot = new Plot();
        Float[] loss = lossList.toArray(new Float[0]);
        plot.line(Util.toFloat(Util.getSeq(lossList.size())), Util.toFloat(loss), "loss");
        plot.show();
    }
}
