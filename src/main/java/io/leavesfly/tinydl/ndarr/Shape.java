package io.leavesfly.tinydl.ndarr;

import io.leavesfly.tinydl.utils.Util;

import java.util.Arrays;
import java.util.Objects;

/**
 * 表示矩阵或向量等N维数组的形状
 */

public class Shape {

    public int[] dimension;

    public int[] multipliers;

    public Shape(int row, int column) {
        int[] _dimension = new int[2];
        _dimension[0] = row;
        _dimension[1] = column;
        init(_dimension);
    }

    public Shape(int... _dimension) {
        init(_dimension);
    }

    private void init(int... _dimension) {
        dimension = _dimension.clone();
        multipliers = new int[_dimension.length];
        int accumulator = 1;
        for (int i = _dimension.length - 1; i >= 0; i--) {
            multipliers[i] = accumulator;
            accumulator *= _dimension[i];
        }
    }

    public int getRow() {
        return dimension[0];
    }

    public int getColumn() {
        return dimension[1];
    }

    /**
     * 是否是矩阵
     *
     * @return
     */
    public boolean isMatrix() {
        return dimension.length == 2;
    }

    /**
     * 对应形状的N维数组的size
     *
     * @return
     */
    public int size() {
        int size = 1;
        for (int dim : dimension) {
            size *= dim;
        }
        return size;
    }

    public int getIndex(int... indices) {
        int index = 0;
        for (int i = 0; i < indices.length; i++) {
            index += indices[i] * multipliers[i];
        }
        return index;
    }


    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Shape shape = (Shape) o;
        return Arrays.equals(dimension, shape.dimension);
    }

    @Override
    public int hashCode() {
        Object[] _dimension = Util.toIntObject(dimension);
        return Objects.hash(_dimension);
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        for (int i : dimension) {
            stringBuilder.append(i).append(",");
        }
        stringBuilder.deleteCharAt(stringBuilder.length() - 1);
        return "[" + stringBuilder + ']';
    }
}