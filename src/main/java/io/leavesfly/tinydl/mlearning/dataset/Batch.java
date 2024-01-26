package io.leavesfly.tinydl.mlearning.dataset;

import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.func.Variable;

/**
 * 一批数据
 */
public class Batch {
    private NdArray[] x;
    private NdArray[] y;

    private int index = 0;

    public Batch(NdArray[] x, NdArray[] y) {
        this.x = x;
        this.y = y;
    }

    public NdArray[] getX() {
        return x;
    }

    public void setX(NdArray[] x) {
        this.x = x;
    }

    public NdArray[] getY() {
        return y;
    }

    public void setY(NdArray[] y) {
        this.y = y;
    }

    public int getSize() {
        return x.length;
    }

    public Variable toVariableX() {
        return new Variable(NdArray.merge(0, x));
    }

    public Variable toVariableY() {
        return new Variable(NdArray.merge(0, y));
    }

    public Pair<NdArray, NdArray> next() {
        if (index >= getSize()) {
            return null;
        }
        Pair<NdArray, NdArray> pair = new Pair<NdArray, NdArray>(x[index], y[index]);
        index++;
        return pair;
    }

    public static class Pair<K, V> {
        public K key;
        public V value;

        public Pair(K key, V value) {
            this.key = key;
            this.value = value;
        }
    }

}
