package io.leavesfly.tinydl.example.cv;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.modality.cv.SimpleConvNet;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;

/**
 * SimpleConvNet 测试示例
 * 验证深度卷积网络的前向传播功能
 */
public class SimpleConvNetTest {
    
    public static void main(String[] args) {
        System.out.println("=== SimpleConvNet 深度卷积网络测试 ===");
        
        // 测试MNIST网络
        testMnistConvNet();
        
        // 测试CIFAR-10网络
        testCifar10ConvNet();
        
        // 测试自定义网络
        testCustomConvNet();
        
        System.out.println("=== 所有测试完成 ===");
    }
    
    /**
     * 测试MNIST卷积网络
     */
    public static void testMnistConvNet() {
        System.out.println("\n--- 测试MNIST卷积网络 ---");
        
        try {
            // 构建MNIST网络
            SimpleConvNet convNet = SimpleConvNet.buildMnistConvNet();
            System.out.println("✓ MNIST网络构建成功");
            
            // 创建模拟输入数据 (batch_size=2, channels=1, height=28, width=28)
            Shape inputShape = new Shape(2, 1, 28, 28);
            NdArray inputData = NdArray.likeRandomN(inputShape);
            Variable input = new Variable(inputData);
            
            System.out.println("输入形状: " + input.getValue().shape.toString());
            
            // 前向传播
            Variable output = convNet.layerForward(input);
            System.out.println("输出形状: " + output.getValue().shape.toString());
            System.out.println("✓ MNIST网络前向传播成功");
            
            // 验证输出形状
            Shape expectedOutput = new Shape(2, 10);  // batch_size=2, num_classes=10
            if (output.getValue().shape.toString().equals(expectedOutput.toString())) {
                System.out.println("✓ 输出形状验证通过");
            } else {
                System.out.println("✗ 输出形状不匹配，期望: " + expectedOutput.toString());
            }
            
        } catch (Exception e) {
            System.out.println("✗ MNIST网络测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试CIFAR-10卷积网络
     */
    public static void testCifar10ConvNet() {
        System.out.println("\n--- 测试CIFAR-10卷积网络 ---");
        
        try {
            // 构建CIFAR-10网络
            SimpleConvNet convNet = SimpleConvNet.buildCifar10ConvNet();
            System.out.println("✓ CIFAR-10网络构建成功");
            
            // 创建模拟输入数据 (batch_size=2, channels=3, height=32, width=32)
            Shape inputShape = new Shape(2, 3, 32, 32);
            NdArray inputData = NdArray.likeRandomN(inputShape);
            Variable input = new Variable(inputData);
            
            System.out.println("输入形状: " + input.getValue().shape.toString());
            
            // 前向传播
            Variable output = convNet.layerForward(input);
            System.out.println("输出形状: " + output.getValue().shape.toString());
            System.out.println("✓ CIFAR-10网络前向传播成功");
            
            // 验证输出形状
            Shape expectedOutput = new Shape(2, 10);  // batch_size=2, num_classes=10
            if (output.getValue().shape.toString().equals(expectedOutput.toString())) {
                System.out.println("✓ 输出形状验证通过");
            } else {
                System.out.println("✗ 输出形状不匹配，期望: " + expectedOutput.toString());
            }
            
        } catch (Exception e) {
            System.out.println("✗ CIFAR-10网络测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试自定义卷积网络
     */
    public static void testCustomConvNet() {
        System.out.println("\n--- 测试自定义卷积网络 ---");
        
        try {
            // 构建自定义网络 (64x64 RGB图像，5个类别)
            SimpleConvNet convNet = SimpleConvNet.buildCustomConvNet(
                "CustomNet", 3, 64, 64, 5);
            System.out.println("✓ 自定义网络构建成功");
            
            // 创建模拟输入数据 (batch_size=1, channels=3, height=64, width=64)
            Shape inputShape = new Shape(1, 3, 64, 64);
            NdArray inputData = NdArray.likeRandomN(inputShape);
            Variable input = new Variable(inputData);
            
            System.out.println("输入形状: " + input.getValue().shape.toString());
            
            // 前向传播
            Variable output = convNet.layerForward(input);
            System.out.println("输出形状: " + output.getValue().shape.toString());
            System.out.println("✓ 自定义网络前向传播成功");
            
            // 验证输出形状
            Shape expectedOutput = new Shape(1, 5);  // batch_size=1, num_classes=5
            if (output.getValue().shape.toString().equals(expectedOutput.toString())) {
                System.out.println("✓ 输出形状验证通过");
            } else {
                System.out.println("✗ 输出形状不匹配，期望: " + expectedOutput.toString());
            }
            
        } catch (Exception e) {
            System.out.println("✗ 自定义网络测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}