package io.leavesfly.tinydl.ndarr;

import io.leavesfly.tinydl.utils.Util;

import java.util.Arrays;
import java.util.Objects;

/**
 * 表示矩阵或向量的形状
 */

public class Shape {

    public int[] dimension;

    public Shape(int row, int column) {
        dimension = new int[2];
        dimension[0] = row;
        dimension[1] = column;
    }

    public Shape(int... _dimension) {
        dimension = _dimension;
    }

    public int getRow() {
        return dimension[0];
    }

    public int getColumn() {
        return dimension[1];
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