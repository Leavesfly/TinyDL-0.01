package io.leavesfly.tinydl.ndarr;

import io.leavesfly.tinydl.utils.Util;

import java.util.Arrays;
import java.util.Objects;

/**
 * 表示矩阵或向量的形状
 */

public class Shape {
    /**
     * 表示多少行
     */
    public int row = 1;
    /**
     * 表示多少列
     */
    public int column = 1;

    public int[] dimension;

    public Shape(int row, int column) {
        this.row = row;
        this.column = column;
        dimension = new int[2];
        dimension[0] = row;
        dimension[1] = column;
    }

    public Shape(int... _dimension) {
        this.row = _dimension[0];
        this.column = _dimension[1];
        dimension = _dimension;
    }

    public int getRow() {
        return row;
    }

    public int getColumn() {
        return column;
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
        return "Shape[" + stringBuilder + ']';
    }
}