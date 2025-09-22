package io.leavesfly.tinydl.mlearning.inference;

import io.leavesfly.tinydl.ndarr.NdArray;

/**
 * 图像数据转换器
 * 
 * 用于图像数据与NdArray之间的相互转换。
 * TODO: 当前实现尚未完成，需要进一步完善图像数据处理逻辑。
 * 
 * @author TinyDL
 * @version 1.0
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