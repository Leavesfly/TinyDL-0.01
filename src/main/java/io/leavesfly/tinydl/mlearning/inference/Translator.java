package io.leavesfly.tinydl.mlearning.inference;

import io.leavesfly.tinydl.ndarr.NdArray;


public interface Translator<I, O> {

    NdArray input2NdArray(I input);

    O ndArray2Output(NdArray ndArray);

}
