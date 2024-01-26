package io.leavesfly.tinydl.mlearning.dataset;

import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.utils.Util;

import java.util.*;

/**
 * 能全量装载到内存的DataSet的简单实现
 */
public abstract class ArrayDataset extends DataSet {

    protected NdArray[] xs;
    protected NdArray[] ys;

    public ArrayDataset(int batchSize) {
        super(batchSize);
    }


    @Override
    public List<Batch> getBatches() {

        int size = xs.length;
        // todo 最后不够一批的直接丢掉
        NdArray[][] ndArrayXs = new NdArray[size / batchSize][batchSize];
        NdArray[][] ndArrayYs = new NdArray[size / batchSize][batchSize];

        List<Batch> batches = new ArrayList<>();
        for (int i = 0; i < ndArrayXs.length; i++) {
            for (int j = 0; j < ndArrayXs[0].length; j++) {
                ndArrayXs[i][j] = xs[i * j + j];
                ndArrayYs[i][j] = ys[i * j + j];
            }
        }
        for (int i = 0; i < size / batchSize; i++) {
            batches.add(new Batch(ndArrayXs[i], ndArrayYs[i]));
        }
        return batches;
    }

    @Override
    public Map<String, DataSet> splitDataset(float trainRatio, float testRatio, float validaRation) {

        if (trainRatio + testRatio + validaRation != 1.0f) {
            throw new RuntimeException("splitDataset parameters error! ");
        }

        if (!Objects.isNull(splitDatasetMap) && !splitDatasetMap.isEmpty()) {
            return splitDatasetMap;
        }

        int start = 0;
        int end = (int) (xs.length * trainRatio);
        NdArray[] _xs = Arrays.copyOfRange(xs, start, end);
        NdArray[] _ys = Arrays.copyOfRange(ys, start, end);
        DataSet dataSet = build(batchSize, _xs, _ys);
        splitDatasetMap.put(Usage.TRAIN.name(), dataSet);

        start = end;
        end = (int) (xs.length * testRatio);
        _xs = Arrays.copyOfRange(xs, start, end);
        _ys = Arrays.copyOfRange(ys, start, end);
        dataSet = build(batchSize, _xs, _ys);
        splitDatasetMap.put(Usage.TEST.name(), dataSet);

        start = end;
        end = (int) (xs.length * validaRation);
        _xs = Arrays.copyOfRange(xs, start, end);
        _ys = Arrays.copyOfRange(ys, start, end);
        dataSet = build(batchSize, _xs, _ys);
        splitDatasetMap.put(Usage.VALIDATION.name(), dataSet);

        return splitDatasetMap;
    }

    @Override
    public void shuffle() {
        int size = xs.length;
        Integer[] shuffleIndex = Util.getSeqIndex(size);
        Collections.shuffle(Arrays.asList(shuffleIndex));

        NdArray[] _xs = new NdArray[size];
        NdArray[] _ys = new NdArray[size];
        for (int i = 0; i < size; i++) {
            _xs[i] = xs[shuffleIndex[i]];
            _ys[i] = ys[shuffleIndex[i]];
        }
        xs = _xs;
        ys = _ys;
    }

    protected abstract DataSet build(int batchSize, NdArray[] xs, NdArray[] ys);

    @Override
    public int getSize() {
        return xs.length;
    }


    public NdArray[] getXs() {
        return xs;
    }

    public NdArray[] getYs() {
        return ys;
    }

    public void setXs(NdArray[] xs) {
        this.xs = xs;
    }

    public void setYs(NdArray[] ys) {
        this.ys = ys;
    }
}
