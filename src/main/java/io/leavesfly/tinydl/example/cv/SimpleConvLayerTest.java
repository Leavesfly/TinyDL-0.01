package io.leavesfly.tinydl.example.cv;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.layer.cnn.ConvLayer;

/**
 * 简单的卷积层测试
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * 用于测试卷积层的基本功能，包括：
 * 1. 卷积层的创建
 * 2. 输入输出形状的验证
 * 3. 前向传播功能的验证
 */
public class SimpleConvLayerTest {
    
    /**
     * 主函数，执行卷积层测试
     * 
     * @param args 命令行参数
     */
    public static void main(String[] args) {
        System.out.println("=== ConvLayer 简单测试 ===");
        
        try {
            // 创建简单的卷积层
            // 输入: [1, 1, 5, 5], 输出: [1, 2, 3, 3] (使用3x3卷积核, stride=1, pad=0)
            Shape inputShape = new Shape(1, 1, 5, 5);
            ConvLayer convLayer = new ConvLayer("test_conv", inputShape, 2, 3, 3, 1, 0);
            
            System.out.println("✓ 卷积层创建成功");
            System.out.println("输入形状: " + inputShape.toString());
            System.out.println("输出形状: " + convLayer.getOutputShape().toString());
            
            // 创建测试输入
            NdArray inputData = NdArray.ones(inputShape);  // 全1的简单测试数据
            Variable input = new Variable(inputData);
            
            // 前向传播
            Variable output = convLayer.layerForward(input);
            System.out.println("✓ 前向传播成功");
            System.out.println("实际输出形状: " + output.getValue().shape.toString());
            
            // 验证形状
            Shape expectedShape = new Shape(1, 2, 3, 3);
            if (output.getValue().shape.toString().equals(expectedShape.toString())) {
                System.out.println("✓ 输出形状验证通过");
            } else {
                System.out.println("✗ 输出形状不匹配");
                System.out.println("期望: " + expectedShape.toString());
                System.out.println("实际: " + output.getValue().shape.toString());
            }
            
        } catch (Exception e) {
            System.out.println("✗ 测试失败: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println("=== 测试完成 ===");
    }
}