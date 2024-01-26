package io.leavesfly.tinydl.mlearning.dataset.simple;

import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.mlearning.dataset.ArrayDataset;
import io.leavesfly.tinydl.mlearning.dataset.DataSet;

import java.util.Random;

public class SinDataSet extends ArrayDataset {
    public SinDataSet(int batchSize) {
        super(batchSize);
    }

    @Override
    protected DataSet build(int batchSize, NdArray[] _xs, NdArray[] _ys) {
        ArrayDataset dataSet = new SinDataSet(batchSize);
        dataSet.setXs(_xs);
        dataSet.setYs(_ys);
        return dataSet;
    }

    @Override
    public void doPrepare() {
        int size = 100;
        Random random = new Random(0);
        xs = new NdArray[size];
        ys = new NdArray[size];
        for (int i = 0; i < size; i++) {
            float random1 = random.nextFloat();
            xs[i] = new NdArray(random1);
            ys[i] = new NdArray(Math.sin(random1 * Math.PI * 2) + random.nextFloat());
        }
    }
}
