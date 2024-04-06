package io.leavesfly.tinydl.ndarr;

import java.util.Arrays;

public class NdArrayUtil {
    /**
     * 按照指定维度对多个矩阵进行叠加
     * axis=0 按行
     * axis=1 按列
     *
     * @param ndArrays
     * @return
     */
    public static NdArray merge(int axis, NdArray... ndArrays) {

        NdArray one = ndArrays[0];
        int[] dimension = Arrays.copyOf(one.getShape().dimension, one.getShape().dimension.length);

        if (axis == 1) {
            dimension[dimension.length - 1] = dimension[dimension.length - 1] * ndArrays.length;
            NdArray ndArray = new NdArray(new Shape(dimension));

            int level = one.getShape().size() / one.getShape().dimension[one.getShape().dimension.length - 1];
            int index = 0;
            for (int i = 0; i < level; i++) {
                for (NdArray array : ndArrays) {
                    int horizontal = array.shape.dimension[array.shape.dimension.length - 1];
                    for (int j = 0; j < horizontal; j++) {
                        ndArray.buffer[index] = array.buffer[j + i * horizontal];
                        index++;
                    }
                }
            }
            return ndArray;

        } else if (axis == 0) {
            dimension[0] = dimension[0] * ndArrays.length;
            NdArray ndArray = new NdArray(new Shape(dimension));

            int index = 0;
            for (NdArray array : ndArrays) {
                for (int j = 0; j < array.buffer.length; j++) {
                    ndArray.buffer[index] = array.buffer[j];
                    index++;
                }
            }
            return ndArray;
        }
        throw new RuntimeException("not impl!");
    }
}


