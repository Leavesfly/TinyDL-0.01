package io.leavesfly.tinydl.mlearning.dataset;

import java.util.List;
import java.util.Map;

// todo 一次无法装载到内存的数据源,只能流式访问
public class StreamDataset extends DataSet {

    public StreamDataset(int batchSize) {
        super(batchSize);
    }

    @Override
    public List<Batch> getBatches() {
        return null;
    }

    @Override
    public void doPrepare() {

    }

    @Override
    public void shuffle() {

    }

    @Override
    public Map<String, DataSet> splitDataset(float trainRatio, float testRatio, float validaRation) {
        return null;
    }

    @Override
    public int getSize() {
        return 0;
    }
}
