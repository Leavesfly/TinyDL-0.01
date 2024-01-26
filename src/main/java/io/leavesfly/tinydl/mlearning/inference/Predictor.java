package io.leavesfly.tinydl.mlearning.inference;

import io.leavesfly.tinydl.utils.Config;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.mlearning.Model;

public class Predictor<I, O> {

    private Translator<I, O> translator;

    private Model model;

    public Predictor(Translator<I, O> translator, Model model) {
        this.translator = translator;
        this.model = model;
    }

    public O predict(I input) {

        NdArray _input = translator.input2NdArray(input);
        Config.train = false;

        Variable _output = model.forward(new Variable(_input));

        return translator.ndArray2Output(_output.getValue());

    }
}
