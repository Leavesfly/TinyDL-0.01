package io.leavesfly.tinydl.utils;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;

public class Util {

    /**
     * 数字微分函数的通用求导
     */
    public static NdArray numericalDiff(Function func, NdArray input, float eps) {

        if (eps == 0) {
            eps = 0.00001F;
        }

        NdArray x1 = input.add(new NdArray(eps));
        NdArray x0 = input.sub(new NdArray(eps));

        NdArray y1 = func.forward(x1);
        NdArray y0 = func.forward(x0);

        return y1.sub(y0).divNum(eps * 2);
    }


    public static int[] getSeq(int size) {
        int[] seq = new int[size];
        for (int i = 0; i < size; i++) {
            seq[i] = i;
        }
        return seq;
    }

    public static Integer[] getSeqIndex(int size) {
        Integer[] seq = new Integer[size];
        for (int i = 0; i < size; i++) {
            seq[i] = i;
        }
        return seq;
    }

    public static int[] toInt(float[] src) {
        int[] res = new int[src.length];
        for (int i = 0; i < src.length; i++) {
            res[i] = (int) src[i];
        }
        return res;
    }

    public static Integer[] toIntObject(int[] src) {
        Integer[] res = new Integer[src.length];
        for (int i = 0; i < src.length; i++) {
            res[i] = src[i];
        }
        return res;
    }

    public static float[] toFloat(int[] src) {
        float[] res = new float[src.length];
        for (int i = 0; i < src.length; i++) {
            res[i] = src[i];
        }
        return res;
    }

    public static float[] toFloat(Float[] src) {
        float[] res = new float[src.length];
        for (int i = 0; i < src.length; i++) {
            res[i] = src[i];
        }
        return res;
    }

    public static String format(float num) {
        return String.format("%,.7g", num);
    }

    /**
     * 返回最大值的索引
     *
     * @param array
     * @return
     */
    public static int argMax(float[] array) {
        float maxValue = Float.MIN_VALUE;
        int maxIndex = 0;
        for (int i = 0; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}
