package io.leavesfly.tinydl.mlearning.dataset;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 表示机器学习中的 用于训练 测试 或者验证的 数据集
 */
public abstract class DataSet {

    protected int batchSize;

    protected Map<String, DataSet> splitDatasetMap = new HashMap<>();

    private boolean hadPrepared = false;

    public DataSet(int batchSize) {
        this.batchSize = batchSize;
    }

    public abstract List<Batch> getBatches();


    public void prepare() {
        if (!hadPrepared) {
            doPrepare();
        }
        hadPrepared = true;
    }

    public abstract void doPrepare();

    /**
     * 将数据随机打散
     */
    public abstract void shuffle();

    /**
     * 将数据随机拆分成 训练 测试和验证集 三部分
     */
    public abstract Map<String, DataSet> splitDataset(float trainRatio, float testRatio, float validaRation);

    public DataSet getTrainDataSet() {
        return splitDatasetMap.get(Usage.TRAIN.name());
    }

    public DataSet getTestDataSet() {
        return splitDatasetMap.get(Usage.TEST.name());

    }

    public DataSet getValidationDataSet() {
        return splitDatasetMap.get(Usage.VALIDATION.name());
    }

    public abstract int getSize();

    public static enum Usage {
        TRAIN, TEST, VALIDATION
    }
}
