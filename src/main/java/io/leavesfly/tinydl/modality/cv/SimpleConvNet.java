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
 * 深度卷积神经网络实现
 * 包含多个卷积层、池化层、全连接层和正则化层的深度架构
 * 适用于图像分类等计算机视觉任务
 */
public class SimpleConvNet extends SequentialBlock {
    
    /**
     * 构造函数
     * @param _name 网络名称
     * @param _xInputShape 输入形状，通常为 [batch_size, channels, height, width]
     * @param _yOutputShape 输出形状，通常为 [batch_size, num_classes]
     */
    public SimpleConvNet(String _name, Shape _xInputShape, Shape _yOutputShape) {
        super(_name, _xInputShape, _yOutputShape);
        // 自动构建深度卷积网络
        buildDeepConvNet();
    }
    
    /**
     * 构建深度卷积网络架构
     * 网络结构：
     * Block1: Conv(32, 3x3) -> ReLU -> Conv(32, 3x3) -> ReLU -> MaxPool(2x2) -> Dropout(0.25)
     * Block2: Conv(64, 3x3) -> ReLU -> Conv(64, 3x3) -> ReLU -> MaxPool(2x2) -> Dropout(0.25)
     * Block3: Conv(128, 3x3) -> ReLU -> Conv(128, 3x3) -> ReLU -> MaxPool(2x2) -> Dropout(0.25)
     * Classifier: Flatten -> FC(512) -> ReLU -> Dropout(0.5) -> FC(num_classes)
     */
    private void buildDeepConvNet() {
        Shape currentShape = inputShape;
        
        // 第一个卷积块 (32 filters)
        currentShape = addConvBlock(currentShape, 32, 3, 1, 1, "block1");
        
        // 第二个卷积块 (64 filters)
        currentShape = addConvBlock(currentShape, 64, 3, 1, 1, "block2");
        
        // 第三个卷积块 (128 filters)
        currentShape = addConvBlock(currentShape, 128, 3, 1, 1, "block3");
        
        // 分类器部分
        addClassifier(currentShape);
        
        // 初始化所有层
        init();
    }
    
    /**
     * 添加卷积块
     * @param inputShape 输入形状
     * @param filterNum 卷积核数量
     * @param filterSize 卷积核大小
     * @param stride 步长
     * @param pad 填充
     * @param blockName 块名称
     * @return 输出形状
     */
    private Shape addConvBlock(Shape inputShape, int filterNum, int filterSize, int stride, int pad, String blockName) {
        // 第一个卷积层
        Layer conv1 = new ConvLayer(blockName + "_conv1", inputShape, filterNum, filterSize, filterSize, stride, pad);
        addLayer(conv1);
        
        // 第一个ReLU激活
        Layer relu1 = new ReLuLayer(blockName + "_relu1", conv1.getOutputShape());
        addLayer(relu1);
        
        // 第二个卷积层（相同参数）
        Layer conv2 = new ConvLayer(blockName + "_conv2", relu1.getOutputShape(), filterNum, filterSize, filterSize, stride, pad);
        addLayer(conv2);
        
        // 第二个ReLU激活
        Layer relu2 = new ReLuLayer(blockName + "_relu2", conv2.getOutputShape());
        addLayer(relu2);
        
        // 最大池化层 (2x2)
        Layer pool = new PoolingLayer(blockName + "_pool", relu2.getOutputShape(), 2, 2, 2, 0);
        addLayer(pool);
        
        // Dropout正则化
        Layer dropout = new Dropout(blockName + "_dropout", 0.25f, pool.getOutputShape());
        addLayer(dropout);
        
        return dropout.getOutputShape();
    }
    
    /**
     * 添加分类器部分
     * @param inputShape 从卷积部分输出的形状
     */
    private void addClassifier(Shape inputShape) {
        // 展平层
        Layer flatten = new FlattenLayer("flatten", inputShape, null);
        addLayer(flatten);
        
        // 第一个全连接层 (512 units)
        Layer fc1 = new AffineLayer("fc1", flatten.getOutputShape(), 512, true);
        addLayer(fc1);
        
        // ReLU激活
        Layer reluFc1 = new ReLuLayer("relu_fc1", fc1.getOutputShape());
        addLayer(reluFc1);
        
        // Dropout正则化
        Layer dropoutFc1 = new Dropout("dropout_fc1", 0.5f, reluFc1.getOutputShape());
        addLayer(dropoutFc1);
        
        // 输出层
        int numClasses = outputShape.dimension[outputShape.dimension.length - 1];
        Layer fcOut = new AffineLayer("fc_out", dropoutFc1.getOutputShape(), numClasses, true);
        addLayer(fcOut);
    }
    
    /**
     * 创建用于MNIST数据集的深度卷积网络
     * 输入：28x28x1，输出：10个类别
     * @return 构建好的卷积网络
     */
    public static SimpleConvNet buildMnistConvNet() {
        Shape inputShape = new Shape(1, 1, 28, 28);  // [batch_size, channels, height, width]
        Shape outputShape = new Shape(1, 10);        // [batch_size, num_classes]
        return new SimpleConvNet("MnistConvNet", inputShape, outputShape);
    }
    
    /**
     * 创建用于CIFAR-10数据集的深度卷积网络
     * 输入：32x32x3，输出：10个类别
     * @return 构建好的卷积网络
     */
    public static SimpleConvNet buildCifar10ConvNet() {
        Shape inputShape = new Shape(1, 3, 32, 32);  // [batch_size, channels, height, width]
        Shape outputShape = new Shape(1, 10);        // [batch_size, num_classes]
        return new SimpleConvNet("Cifar10ConvNet", inputShape, outputShape);
    }
    
    /**
     * 创建自定义的深度卷积网络
     * @param name 网络名称
     * @param channels 输入通道数
     * @param height 输入高度
     * @param width 输入宽度
     * @param numClasses 输出类别数
     * @return 构建好的卷积网络
     */
    public static SimpleConvNet buildCustomConvNet(String name, int channels, int height, int width, int numClasses) {
        Shape inputShape = new Shape(1, channels, height, width);
        Shape outputShape = new Shape(1, numClasses);
        return new SimpleConvNet(name, inputShape, outputShape);
    }
    
    /**
     * 保持向后兼容的静态方法
     * @return 构建好的卷积网络
     * @deprecated 建议使用具体的构建方法如 buildMnistConvNet()
     */
    @Deprecated
    public static SequentialBlock builtConvNet() {
        return buildMnistConvNet();
    }
}
