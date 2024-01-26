package io.leavesfly.tinydl.nnet;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.Shape;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 表示由层组合起来的更大的神经网络的块
 */
public abstract class Block implements LayerAble {

    protected String name;

    protected Map<String, Parameter> params;

    protected Shape xInputShape;

    protected Shape yOutputShape;

    protected List<LayerAble> layers;

    public Block(String _name, Shape _xInputShape, Shape _yOutputShape) {
        name = _name;
        this.params = new HashMap<>();
        layers = new ArrayList<>();
        xInputShape = _xInputShape;
        yOutputShape = _yOutputShape;
    }

    @Override
    public void clearGrads() {
        for (Parameter parameter : params.values()) {
            parameter.clearGrad();
        }
        for (LayerAble layer : layers) {
            layer.clearGrads();
        }
    }

    public void addLayer(Layer layer) {
        layer.init();
        getLayers().add(layer);
    }



    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];
        Variable y = layers.get(0).forward(x);
        for (int i = 1; i < layers.size(); i++) {
            y = layers.get(i).forward(y);
        }
        return y;
    }

    public Map<String, Parameter> getAllParams() {
        Map<String, Parameter> allParams = new HashMap<>();
        putAll(allParams);
        return allParams;
    }

    private void putAll(Map<String, Parameter> allParams) {
        allParams.putAll(params);
        for (LayerAble layer : layers) {
            allParams.putAll(layer.getParams());
        }
    }

    public void resetState() {
        for (LayerAble layerAble : layers) {
            if (layerAble instanceof RnnLayer) {
                ((RnnLayer) layerAble).resetState();
            } else if (layerAble instanceof Block) {
                ((Block) layerAble).resetState();
            }
        }
    }

    @Override
    public Shape getXInputShape() {
        return xInputShape;
    }

    @Override
    public Shape getYOutputShape() {
        return yOutputShape;
    }

    public String getName() {
        return name;
    }

    public Map<String, Parameter> getParams() {
        return params;
    }

    public void addParam(String paramName, Parameter value) {
        getParams().put(name + "." + paramName, value);
    }

    public Parameter getParamBy(String paramName) {
        return getParams().get(name + "." + paramName);
    }

    public List<LayerAble> getLayers() {
        return layers;
    }

}
