package io.leavesfly.tinydl.example.regress;

import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.utils.Plot;
import io.leavesfly.tinydl.utils.Uml;

import java.util.Random;

/**
 * 带有噪声的线性数据的拟合
 */
public class LineExam {
    public static void main(String[] args) {

        //====== 1，生成数据====
        Random random = new Random(0);
        float[] x = new float[100];
        for (int i = 0; i < x.length; i++) {
            x[i] = random.nextFloat();
        }

        float[] y = new float[100];
        for (int i = 0; i < y.length; i++) {
            y[i] = 5 + 2 * x[i] + random.nextFloat();
        }

        Variable w = new Variable(new NdArray(0), "w");
        Variable b = new Variable(new NdArray(0), "b");

        Variable variableX = new Variable(new NdArray(x), "x", false).transpose();
        Variable variableY = new Variable(new NdArray(y), "y", false).transpose();

        NdArray learnRate = new NdArray(0.1f);
        int maxEpoch = 100;

        //====== 2，训练模型====
        Variable lastLoss = null;
        for (int i = 0; i < maxEpoch; i++) {
            Variable predict = predict(variableX, w, b);
            Variable loss = meanSquaError(variableY, predict);

            w.clearGrad();
            b.clearGrad();

            loss.setName("loss");
            //反向梯度传播
            loss.backward();

            w.setValue(w.getValue().sub(w.getGrad().mul(learnRate)));
            b.setValue(b.getValue().sub(b.getGrad().mul(learnRate)));

            if (i % (maxEpoch / 10) == 0 || (i == maxEpoch - 1)) {
                System.out.println("i=" + i + " w:" + w.getValue().getNumber() + " b:" + b.getValue().getNumber()
                        + " loss:" + loss.getValue().getNumber().floatValue());
            }
            lastLoss = loss;
        }

        System.out.println(Uml.getDotGraph(lastLoss));

        Variable predictY = predict(variableX, w, b);
        float[] p_y = predictY.transpose().getValue().getMatrix()[0];
        //画图
        Plot plot = new Plot();
        plot.scatter(x, y);
        plot.line(x, p_y, "line");
        plot.show();
    }

    public static Variable predict(Variable x, Variable w, Variable b) {
        return x.linear(w, b);
    }


    public static Variable meanSquaError(Variable y, Variable x) {
        return y.sub(x).squ().sum().div(new Variable(y.getValue().getMatrix().length));
    }
}
