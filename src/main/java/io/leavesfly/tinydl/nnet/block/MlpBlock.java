package io.leavesfly.tinydl.nnet.block;

import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Block;
import io.leavesfly.tinydl.nnet.layer.activate.ReLuLayer;
import io.leavesfly.tinydl.nnet.layer.activate.SigmoidLayer;
import io.leavesfly.tinydl.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinydl.utils.Config;

import io.leavesfly.tinydl.nnet.Layer;

import java.util.Objects;

/**
 * 多层全连接的BP神经网络
 */
public class MlpBlock extends Block {
    Config.ActiveFunc activeFunc;

    public MlpBlock(String _name, int batchSize, Config.ActiveFunc _activeFunc, int... layerSizes) {
        super(_name, new Shape(batchSize, layerSizes[0]), new Shape(-1, layerSizes[layerSizes.length - 1]));

        activeFunc = _activeFunc;

        for (int i = 1; i < layerSizes.length - 1; i++) {
            Layer layer = new LinearLayer("layer" + i, layerSizes[i - 1], layerSizes[i], true);
            addLayer(layer);
            if (!Objects.isNull(activeFunc) && Config.ActiveFunc.ReLU.name().equals(activeFunc.name())) {
                addLayer(new ReLuLayer("ReLU"));
            } else {
                addLayer(new SigmoidLayer("Sigmoid"));
            }
        }
        Layer layer = new LinearLayer("layer" + (layerSizes.length - 1), layerSizes[(layerSizes.length - 2)], layerSizes[(layerSizes.length - 1)], true);
        addLayer(layer);
    }

    @Override
    public void init() {

    }
}
