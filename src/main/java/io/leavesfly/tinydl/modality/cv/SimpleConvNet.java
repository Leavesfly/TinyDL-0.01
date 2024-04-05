package io.leavesfly.tinydl.modality.cv;

import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Layer;
import io.leavesfly.tinydl.nnet.block.SequentialBlock;
import io.leavesfly.tinydl.nnet.layer.activate.ReLuLayer;
import io.leavesfly.tinydl.nnet.layer.cnn.ConvLayer;
import io.leavesfly.tinydl.nnet.layer.cnn.PoolingLayer;
import io.leavesfly.tinydl.nnet.layer.dnn.AffineLayer;
import io.leavesfly.tinydl.nnet.layer.norm.Dropout;
import io.leavesfly.tinydl.nnet.layer.norm.FlattenLayer;

/**
 * todo 深度的卷积网络 待完善
 */
public class SimpleConvNet extends SequentialBlock {
    public SimpleConvNet(String _name, Shape _xInputShape, Shape _yOutputShape) {
        super(_name, _xInputShape, _yOutputShape);
    }

    public static SequentialBlock builtConvNet() {

        Shape inputXShape = new Shape(28, 28, 1);
        Shape outputYShape = new Shape(10);
        SequentialBlock sequentialBlock = new SequentialBlock("ConvNet", inputXShape, outputYShape);

        Layer layer = new ConvLayer("ConvLayer", 5, 32, ConvLayer.PaddingType.SAME, inputXShape, null);
        sequentialBlock.addLayer(layer);

        inputXShape = layer.getYOutputShape();
        layer = new ReLuLayer("ReLuLayer", inputXShape);
        sequentialBlock.addLayer(layer);

        inputXShape = layer.getYOutputShape();
        layer = new PoolingLayer("PoolingLayer", 2, 2, inputXShape, null);
        sequentialBlock.addLayer(layer);

        inputXShape = layer.getYOutputShape();
        layer = new ConvLayer("ConvLayer", 5, 64, ConvLayer.PaddingType.SAME, inputXShape, null);
        sequentialBlock.addLayer(layer);

        inputXShape = layer.getYOutputShape();
        layer = new ReLuLayer("ReLuLayer", inputXShape);
        sequentialBlock.addLayer(layer);

        inputXShape = layer.getYOutputShape();
        layer = new PoolingLayer("PoolingLayer", 2, 2, inputXShape, null);
        sequentialBlock.addLayer(layer);

        inputXShape = layer.getYOutputShape();
        layer = new FlattenLayer("FlattenLayer", inputXShape, null);
        sequentialBlock.addLayer(layer);

        inputXShape = layer.getYOutputShape();
        layer = new AffineLayer("AffineLayer", inputXShape, 1024, true);
        sequentialBlock.addLayer(layer);

        inputXShape = layer.getYOutputShape();
        layer = new ReLuLayer("ReLuLayer", inputXShape);
        sequentialBlock.addLayer(layer);

        inputXShape = layer.getYOutputShape();
        layer = new Dropout("Dropout", 0.3f, inputXShape);
        sequentialBlock.addLayer(layer);


        inputXShape = layer.getYOutputShape();
        layer = new AffineLayer("AffineLayer", inputXShape, 10, true);
        sequentialBlock.addLayer(layer);

        return sequentialBlock;
    }

}
