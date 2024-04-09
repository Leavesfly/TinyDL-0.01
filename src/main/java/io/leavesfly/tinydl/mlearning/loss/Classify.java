package io.leavesfly.tinydl.mlearning.loss;

import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.func.Variable;

public class Classify {
    public float accuracyRate(Variable label, Variable predict) {
        int size = label.getValue().getShape().getRow();
        NdArray labelNdArray = label.getValue();
        NdArray predictNdArray = predict.getValue();

        NdArray argMax = predictNdArray.argMax(1);
        NdArray sames = argMax.eq(labelNdArray);
        return sames.sum().divNum((float) size).getNumber().floatValue();
    }
}
