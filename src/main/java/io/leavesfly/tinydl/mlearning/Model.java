package io.leavesfly.tinydl.mlearning;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.mlearning.inference.Predictor;
import io.leavesfly.tinydl.mlearning.inference.Translator;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Block;
import io.leavesfly.tinydl.nnet.Parameter;
import io.leavesfly.tinydl.utils.Uml;

import java.io.*;
import java.util.Map;

/**
 * 机器模型的表示
 */
public class Model implements Serializable {

    private String name;

    private Block block;

    public transient Variable tmpPredict;

    public Model(String _name, Block _block) {
        name = _name;
        block = _block;
    }

    public void plot() {
        Shape xInputShape = block.getInputShape();
        if (xInputShape != null) {
            Shape shape = block.getInputShape();
            tmpPredict = block.layerForward(new Variable(NdArray.ones(shape)));
        }
        System.out.println(Uml.getDotGraph(tmpPredict));
    }

    public void save(File modelFile) {

        try (FileOutputStream fileOut = new FileOutputStream(modelFile); ObjectOutputStream out = new ObjectOutputStream(fileOut)) {
            out.writeObject(this);
        } catch (Exception e) {
            throw new RuntimeException("Model save error!");
        }
    }

    public static Model load(File modelFile) {
        try (FileInputStream fileIn = new FileInputStream(modelFile); ObjectInputStream in = new ObjectInputStream(fileIn)) {
            return (Model) in.readObject();
        } catch (Exception e) {
            throw new RuntimeException("model load error!");
        }
    }

    public void resetState() {
        block.resetState();
    }

    public Variable forward(Variable... inputs) {

        return block.layerForward(inputs);
    }

    public void clearGrads() {
        block.clearGrads();
    }

    public Map<String, Parameter> getAllParams() {
        return block.getAllParams();
    }

    public <I, O> Predictor<I, O> getPredictor(Translator<I, O> translator) {
        return new Predictor<>(translator, this);
    }

    public String getName() {
        return name;
    }

    public Block getBlock() {
        return block;
    }

}
