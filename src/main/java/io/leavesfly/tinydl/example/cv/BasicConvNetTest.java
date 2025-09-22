package io.leavesfly.tinydl.example.cv;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.block.SequentialBlock;
import io.leavesfly.tinydl.nnet.layer.activate.ReLuLayer;
import io.leavesfly.tinydl.nnet.layer.cnn.ConvLayer;
import io.leavesfly.tinydl.nnet.layer.cnn.PoolingLayer;
import io.leavesfly.tinydl.nnet.layer.dnn.AffineLayer;
import io.leavesfly.tinydl.nnet.layer.norm.FlattenLayer;

/**
 * 简单的卷积网络测试
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * 测试一个基本的卷积网络结构：Conv -> ReLU -> Pool -> Flatten -> FC
 * 用于验证卷积层、池化层、激活层和全连接层的基本功能和形状变换。
 */
public class BasicConvNetTest {
    
    /**
     * 主函数，执行基本卷积网络测试
     * 
     * @param args 命令行参数
     */
    public static void main(String[] args) {
        System.out.println("=== 基本卷积网络测试 ===");
        
        try {
            // 构建一个简单的卷积网络
            Shape inputShape = new Shape(1, 1, 8, 8);  // [batch_size, channels, height, width]
            Shape outputShape = new Shape(1, 2);        // [batch_size, num_classes]
            
            SequentialBlock convNet = new SequentialBlock("BasicConvNet", inputShape, outputShape);
            
            // 卷积层: 1->4 通道, 3x3 卷积核, stride=1, pad=1
            ConvLayer conv = new ConvLayer("conv1", inputShape, 4, 3, 3, 1, 1);
            convNet.addLayer(conv);
            System.out.println("卷积层输出形状: " + conv.getOutputShape().toString());
            
            // ReLU 激活
            ReLuLayer relu = new ReLuLayer("relu1", conv.getOutputShape());
            convNet.addLayer(relu);
            
            // 池化层: 2x2 池化, stride=2
            PoolingLayer pool = new PoolingLayer("pool1", relu.getOutputShape(), 2, 2, 2, 0);
            convNet.addLayer(pool);
            System.out.println("池化层输出形状: " + pool.getOutputShape().toString());
            
            // 展平层
            FlattenLayer flatten = new FlattenLayer("flatten", pool.getOutputShape(), null);
            convNet.addLayer(flatten);
            System.out.println("展平层输出形状: " + flatten.getOutputShape().toString());
            
            // 全连接层
            AffineLayer fc = new AffineLayer("fc1", flatten.getOutputShape(), 2, true);
            convNet.addLayer(fc);
            System.out.println("全连接层输出形状: " + fc.getOutputShape().toString());
            
            System.out.println("✓ 网络构建成功");
            
            // 创建测试输入
            NdArray inputData = NdArray.ones(inputShape);
            Variable input = new Variable(inputData);
            
            System.out.println("输入形状: " + input.getValue().shape.toString());
            
            // 前向传播
            Variable output = convNet.layerForward(input);
            System.out.println("输出形状: " + output.getValue().shape.toString());
            System.out.println("✓ 前向传播成功");
            
            // 验证输出形状
            if (output.getValue().shape.toString().equals(outputShape.toString())) {
                System.out.println("✓ 输出形状验证通过");
            } else {
                System.out.println("✗ 输出形状不匹配");
                System.out.println("期望: " + outputShape.toString());
                System.out.println("实际: " + output.getValue().shape.toString());
            }
            
        } catch (Exception e) {
            System.out.println("✗ 测试失败: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println("=== 测试完成 ===");
    }
}