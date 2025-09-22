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
import io.leavesfly.tinydl.nnet.layer.norm.BatchNormLayer;

/**
 * 增强的深度卷积神经网络实现
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * SimpleConvNet类实现了增强的深度卷积神经网络，包含多个卷积层、池化层、全连接层和正则化层的深度架构。
 * 支持批量归一化、残差连接和灵活配置，适用于图像分类等计算机视觉任务。
 */
public class SimpleConvNet extends SequentialBlock {
    
    /**
     * 网络配置类
     * 
     * 用于配置卷积神经网络的各种超参数，包括批量归一化、残差连接、偏置使用、Dropout率等。
     */
    public static class ConvNetConfig {
        /**
         * 是否使用批量归一化
         */
        public boolean useBatchNorm = true;
        
        /**
         * 是否使用残差连接
         */
        public boolean useResidual = false;
        
        /**
         * 卷积层是否使用偏置
         */
        public boolean useBias = true;
        
        /**
         * Dropout率
         */
        public float dropoutRate = 0.5f;
        
        /**
         * 每个块的滤波器数量
         */
        public int[] filterNums = {32, 64, 128};
        
        /**
         * 卷积核大小
         */
        public int filterSize = 3;
        
        /**
         * 步长
         */
        public int stride = 1;
        
        /**
         * 填充
         */
        public int pad = 1;
        
        /**
         * 全连接层隐藏单元数
         */
        public int fcHiddenSize = 512;
        
        /**
         * 默认构造函数
         */
        public ConvNetConfig() {}
        
        /**
         * 设置是否使用批量归一化
         * 
         * @param use 是否使用批量归一化
         * @return 当前配置对象
         */
        public ConvNetConfig useBatchNorm(boolean use) {
            this.useBatchNorm = use;
            return this;
        }
        
        /**
         * 设置是否使用残差连接
         * 
         * @param use 是否使用残差连接
         * @return 当前配置对象
         */
        public ConvNetConfig useResidual(boolean use) {
            this.useResidual = use;
            return this;
        }
        
        /**
         * 设置Dropout率
         * 
         * @param rate Dropout率
         * @return 当前配置对象
         */
        public ConvNetConfig dropoutRate(float rate) {
            this.dropoutRate = rate;
            return this;
        }
        
        /**
         * 设置每个块的滤波器数量
         * 
         * @param nums 滤波器数量数组
         * @return 当前配置对象
         */
        public ConvNetConfig filterNums(int... nums) {
            this.filterNums = nums;
            return this;
        }
        
        /**
         * 设置全连接层隐藏单元数
         * 
         * @param size 隐藏单元数
         * @return 当前配置对象
         */
        public ConvNetConfig fcHiddenSize(int size) {
            this.fcHiddenSize = size;
            return this;
        }
    }
    
    /**
     * 网络配置实例
     */
    private ConvNetConfig config;
    
    /**
     * 构造函数（使用默认配置）
     * 
     * @param _name 网络名称
     * @param _xInputShape 输入形状，通常为 [batch_size, channels, height, width]
     * @param _yOutputShape 输出形状，通常为 [batch_size, num_classes]
     */
    public SimpleConvNet(String _name, Shape _xInputShape, Shape _yOutputShape) {
        this(_name, _xInputShape, _yOutputShape, new ConvNetConfig());
    }
    
    /**
     * 完整构造函数（使用自定义配置）
     * 
     * @param _name 网络名称
     * @param _xInputShape 输入形状，通常为 [batch_size, channels, height, width]
     * @param _yOutputShape 输出形状，通常为 [batch_size, num_classes]
     * @param _config 网络配置
     */
    public SimpleConvNet(String _name, Shape _xInputShape, Shape _yOutputShape, ConvNetConfig _config) {
        super(_name, _xInputShape, _yOutputShape);
        this.config = _config;
        // 自动构建增强的深度卷积网络
        buildEnhancedConvNet();
    }
    
    /**
     * 构建增强的深度卷积网络架构
     * 
     * 网络结构：
     * - 多个卷积块（可配置数量和参数）
     * - 每个块包含：Conv -> BatchNorm(可选) -> ReLU -> Conv -> BatchNorm(可选) -> ReLU -> MaxPool -> Dropout
     * - 残差连接(可选)
     * - 分类器：Flatten -> FC -> ReLU -> Dropout -> FC(output)
     */
    private void buildEnhancedConvNet() {
        Shape currentShape = inputShape;
        
        // 构建卷积块
        for (int i = 0; i < config.filterNums.length; i++) {
            int filterNum = config.filterNums[i];
            String blockName = "block" + (i + 1);
            
            if (config.useResidual && i > 0 && currentShape.dimension[1] == filterNum) {
                // 残差块
                currentShape = addResidualBlock(currentShape, filterNum, blockName);
            } else {
                // 普通卷积块
                currentShape = addEnhancedConvBlock(currentShape, filterNum, blockName);
            }
        }
        
        // 分类器部分
        addEnhancedClassifier(currentShape);
        
        // 初始化所有层
        init();
    }
    
    /**
     * 添加增强的卷积块
     * 
     * @param inputShape 输入形状
     * @param filterNum 卷积核数量
     * @param blockName 块名称
     * @return 输出形状
     */
    private Shape addEnhancedConvBlock(Shape inputShape, int filterNum, String blockName) {
        // 第一个卷积层
        Layer conv1 = new ConvLayer(blockName + "_conv1", inputShape, filterNum, 
                                   config.filterSize, config.filterSize, config.stride, config.pad, config.useBias);
        addLayer(conv1);
        
        Shape currentShape = conv1.getOutputShape();
        
        // 批量归一化（如果启用）
        if (config.useBatchNorm) {
            Layer bn1 = new BatchNormLayer(blockName + "_bn1", currentShape);
            addLayer(bn1);
            currentShape = bn1.getOutputShape();
        }
        
        // 第一个ReLU激活
        Layer relu1 = new ReLuLayer(blockName + "_relu1", currentShape);
        addLayer(relu1);
        currentShape = relu1.getOutputShape();
        
        // 第二个卷积层
        Layer conv2 = new ConvLayer(blockName + "_conv2", currentShape, filterNum, 
                                   config.filterSize, config.filterSize, config.stride, config.pad, config.useBias);
        addLayer(conv2);
        currentShape = conv2.getOutputShape();
        
        // 第二个批量归一化（如果启用）
        if (config.useBatchNorm) {
            Layer bn2 = new BatchNormLayer(blockName + "_bn2", currentShape);
            addLayer(bn2);
            currentShape = bn2.getOutputShape();
        }
        
        // 第二个ReLU激活
        Layer relu2 = new ReLuLayer(blockName + "_relu2", currentShape);
        addLayer(relu2);
        currentShape = relu2.getOutputShape();
        
        // 最大池化层 (2x2)
        Layer pool = new PoolingLayer(blockName + "_pool", currentShape, 2, 2, 2, 0);
        addLayer(pool);
        currentShape = pool.getOutputShape();
        
        // Dropout正则化
        float dropoutRate = Math.min(config.dropoutRate * 0.5f, 0.25f); // 卷积层使用较低的dropout率
        Layer dropout = new Dropout(blockName + "_dropout", dropoutRate, currentShape);
        addLayer(dropout);
        
        return dropout.getOutputShape();
    }
    
    /**
     * 添加残差块（简化版本）
     * 
     * @param inputShape 输入形状
     * @param filterNum 卷积核数量
     * @param blockName 块名称
     * @return 输出形状
     */
    private Shape addResidualBlock(Shape inputShape, int filterNum, String blockName) {
        // 残差块的简化实现：直接在卷积后添加跳连
        // 注意：这里需要实际的残差连接实现，目前先使用普通块
        return addEnhancedConvBlock(inputShape, filterNum, blockName);
    }
    
    /**
     * 添加增强的分类器部分
     * 
     * @param inputShape 从卷积部分输出的形状
     */
    private void addEnhancedClassifier(Shape inputShape) {
        // 展平层
        Layer flatten = new FlattenLayer("flatten", inputShape, null);
        addLayer(flatten);
        
        // 第一个全连接层
        Layer fc1 = new AffineLayer("fc1", flatten.getOutputShape(), config.fcHiddenSize, true);
        addLayer(fc1);
        
        // ReLU激洿
        Layer reluFc1 = new ReLuLayer("relu_fc1", fc1.getOutputShape());
        addLayer(reluFc1);
        
        // Dropout正则化
        Layer dropoutFc1 = new Dropout("dropout_fc1", config.dropoutRate, reluFc1.getOutputShape());
        addLayer(dropoutFc1);
        
        // 输出层
        int numClasses = outputShape.dimension[outputShape.dimension.length - 1];
        Layer fcOut = new AffineLayer("fc_out", dropoutFc1.getOutputShape(), numClasses, true);
        addLayer(fcOut);
    }
    
    /**
     * 添加卷积块
     * 
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
     * 
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
     * 
     * @return 构建好的卷积网络
     */
    public static SimpleConvNet buildMnistConvNet() {
        Shape inputShape = new Shape(1, 1, 28, 28);  // [batch_size, channels, height, width]
        Shape outputShape = new Shape(1, 10);        // [batch_size, num_classes]
        
        ConvNetConfig config = new ConvNetConfig()
                .filterNums(16, 32, 64)  // 适合MNIST的较小网络
                .dropoutRate(0.25f)
                .fcHiddenSize(128);
                
        return new SimpleConvNet("MnistConvNet", inputShape, outputShape, config);
    }
    
    /**
     * 创建用于CIFAR-10数据集的深度卷积网络
     * 输入：32x32x3，输出：10个类别
     * 
     * @return 构建好的卷积网络
     */
    public static SimpleConvNet buildCifar10ConvNet() {
        Shape inputShape = new Shape(1, 3, 32, 32);  // [batch_size, channels, height, width]
        Shape outputShape = new Shape(1, 10);        // [batch_size, num_classes]
        
        ConvNetConfig config = new ConvNetConfig()
                .filterNums(32, 64, 128, 256)  // 适合CIFAR-10的中等网络
                .dropoutRate(0.4f)
                .fcHiddenSize(512);
                
        return new SimpleConvNet("Cifar10ConvNet", inputShape, outputShape, config);
    }
    
    /**
     * 创建带残差连接的深度卷积网络
     * 
     * @return 构建好的卷积网络
     */
    public static SimpleConvNet buildResNetStyle() {
        Shape inputShape = new Shape(1, 3, 32, 32);
        Shape outputShape = new Shape(1, 10);
        
        ConvNetConfig config = new ConvNetConfig()
                .filterNums(32, 32, 64, 64, 128, 128)  // 更深的网络
                .useResidual(true)
                .useBatchNorm(true)
                .dropoutRate(0.3f)
                .fcHiddenSize(512);
                
        return new SimpleConvNet("ResNetStyle", inputShape, outputShape, config);
    }
    
    /**
     * 创建自定义的深度卷积网络
     * 
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
     * 创建自定义的深度卷积网络（带配置）
     * 
     * @param name 网络名称
     * @param channels 输入通道数
     * @param height 输入高度
     * @param width 输入宽度
     * @param numClasses 输出类别数
     * @param config 网络配置
     * @return 构建好的卷积网络
     */
    public static SimpleConvNet buildCustomConvNet(String name, int channels, int height, int width, int numClasses, ConvNetConfig config) {
        Shape inputShape = new Shape(1, channels, height, width);
        Shape outputShape = new Shape(1, numClasses);
        return new SimpleConvNet(name, inputShape, outputShape, config);
    }
    
    /**
     * 保持向后兼容的静态方法
     * 
     * @return 构建好的卷积网络
     * @deprecated 建议使用具体的构建方法如 buildMnistConvNet()
     */
    @Deprecated
    public static SequentialBlock builtConvNet() {
        return buildMnistConvNet();
    }
}