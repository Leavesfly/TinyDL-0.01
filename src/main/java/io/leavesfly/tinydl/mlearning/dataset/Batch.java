package io.leavesfly.tinydl.mlearning.dataset;

import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArrayUtil;

/**
 * 一批数据
 * 
 * 该类表示机器学习中的一批训练或测试数据，包含输入数据和对应的标签数据。
 * 提供了将数据转换为Variable对象的方法，便于模型训练和推理。
 * 
 * @author TinyDL
 * @version 1.0
 */
public class Batch {
    private NdArray[] x;
    private NdArray[] y;

    private int index = 0;

    /**
     * 构造函数
     * @param x 输入数据数组
     * @param y 标签数据数组
     */
    public Batch(NdArray[] x, NdArray[] y) {
        this.x = x;
        this.y = y;
    }

    /**
     * 获取输入数据数组
     * @return 输入数据数组
     */
    public NdArray[] getX() {
        return x;
    }

    /**
     * 设置输入数据数组
     * @param x 输入数据数组
     */
    public void setX(NdArray[] x) {
        this.x = x;
    }

    /**
     * 获取标签数据数组
     * @return 标签数据数组
     */
    public NdArray[] getY() {
        return y;
    }

    /**
     * 设置标签数据数组
     * @param y 标签数据数组
     */
    public void setY(NdArray[] y) {
        this.y = y;
    }

    /**
     * 获取批次大小
     * @return 批次大小
     */
    public int getSize() {
        return x.length;
    }

    /**
     * 将输入数据转换为Variable对象
     * @return 输入数据的Variable表示
     */
    public Variable toVariableX() {
        return new Variable(NdArrayUtil.merge(0, x));
    }

    /**
     * 将标签数据转换为Variable对象
     * @return 标签数据的Variable表示
     */
    public Variable toVariableY() {
        return new Variable(NdArrayUtil.merge(0, y));
    }

    /**
     * 获取下一对数据
     * @return 数据对，如果已遍历完则返回null
     */
    public Pair<NdArray, NdArray> next() {
        if (index >= getSize()) {
            return null;
        }
        Pair<NdArray, NdArray> pair = new Pair<NdArray, NdArray>(x[index], y[index]);
        index++;
        return pair;
    }

    /**
     * 键值对内部类
     * @param <K> 键类型
     * @param <V> 值类型
     */
    public static class Pair<K, V> {
        public K key;
        public V value;

        /**
         * 构造函数
         * @param key 键
         * @param value 值
         */
        public Pair(K key, V value) {
            this.key = key;
            this.value = value;
        }
    }

}