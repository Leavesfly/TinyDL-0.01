package io.leavesfly.tinydl.mlearning.dataset.simple;

import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.mlearning.dataset.ArrayDataset;
import io.leavesfly.tinydl.mlearning.dataset.DataSet;
import io.leavesfly.tinydl.utils.Util;

import java.util.Random;

public class SpiralDateSet extends ArrayDataset {
    public SpiralDateSet(int batchSize) {
        super(batchSize);
    }

    public SpiralDateSet(int batchSize, NdArray[] _xs, NdArray[] _ys) {
        super(batchSize);
        xs = _xs;
        ys = _ys;
    }

    @Override
    public void doPrepare() {

        int num_data = 100;
        int num_class = 3;
        int input_dim = 2;
        int data_size = num_class * num_data;

        xs = new NdArray[data_size];
        ys = new NdArray[data_size];
        Random random = new Random(0);

        for (int j = 0; j < num_class; j++) {
            for (int i = 0; i < num_data; i++) {
                float rate = i / (float) num_data;
                float radius = (float) (1.0 * rate);
                float theta = (float) (j * 4.0 + 4.0 * rate + random.nextGaussian() * 0.2);

                int index = j * num_data + i;
                float[] x = new float[input_dim];
                x[0] = (float) (radius * Math.sin(theta));
                x[1] = (float) (radius * Math.cos(theta));
                xs[index] = new NdArray(x);
                ys[index] = new NdArray((float) j);
            }
        }

        DataSet trainDataset = build(batchSize, xs, ys);
        splitDatasetMap.put(Usage.TRAIN.name(), trainDataset);

        DataSet testDataset = build(batchSize, xs, ys);
        splitDatasetMap.put(Usage.TEST.name(), testDataset);
    }


    @Override
    protected DataSet build(int batchSize, NdArray[] _xs, NdArray[] _ys) {
        ArrayDataset dataSet = new SpiralDateSet(batchSize);
        dataSet.setXs(_xs);
        dataSet.setYs(_ys);
        return dataSet;
    }

    public static SpiralDateSet toSpiralDateSet(Variable x, Variable y) {
        int size = x.getValue().getShape().getRow();
        NdArray[] xs = new NdArray[size];
        NdArray[] ys = new NdArray[size];

        float[][] x_mat = x.getValue().getMatrix();
        float[][] y_mat = y.getValue().getMatrix();

        for (int i = 0; i < size; i++) {
            xs[i] = new NdArray(x_mat[i]);
            ys[i] = new NdArray(Util.argMax(y_mat[i]));
        }
        return new SpiralDateSet(100, xs, ys);
    }
}
