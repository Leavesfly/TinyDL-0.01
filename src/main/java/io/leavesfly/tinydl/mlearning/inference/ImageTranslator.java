package io.leavesfly.tinydl.mlearning.inference;

import io.leavesfly.tinydl.ndarr.NdArray;

/**
 * todo 图像数据格式转化
 */
public class ImageTranslator implements Translator<float[][], String> {
    @Override
    public NdArray input2NdArray(float[][] input) {

        return new NdArray(input);
    }

    @Override
    public String ndArray2Output(NdArray ndArray) {

        return "image :" + ndArray.getNumber().intValue();
    }
}
