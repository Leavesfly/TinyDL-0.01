package io.leavesfly.tinydl.example.cv;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.layer.cnn.ConvLayer;

/**
 * ConvLayer优化示例
 * 
 * 展示优化后的ConvLayer类的功能和使用方法
 */
public class ConvLayerOptimizationExample {
    
    public static void main(String[] args) {
        System.out.println("ConvLayer优化示例");
        System.out.println("==================");
        
        // 测试1: 基本卷积层功能
        System.out.println("测试1: 基本卷积层功能");
        testBasicConvLayer();
        
        // 测试2: 带偏置的卷积层
        System.out.println("\n测试2: 带偏置的卷积层");
        testConvLayerWithBias();
        
        // 测试3: 不同参数的卷积层
        System.out.println("\n测试3: 不同参数的卷积层");
        testDifferentConvParams();
        
        // 测试4: 前向传播
        System.out.println("\n测试4: 前向传播");
        testForwardPropagation();
    }
    
    private static void testBasicConvLayer() {
        try {
            // 创建输入形状 [1, 1, 4, 4]
            Shape inputShape = new Shape(1, 1, 4, 4);
            
            // 创建卷积层：1个卷积核，3x3大小，步长1，无填充
            ConvLayer convLayer = new ConvLayer("basic_conv", inputShape, 1, 3, 3, 1, 0);
            
            System.out.println("  ✓ 卷积层创建成功");
            System.out.println("  输入形状: " + inputShape.toString());
            System.out.println("  输出形状: " + convLayer.getOutputShape().toString());
            System.out.println("  卷积核数量: " + getField(convLayer, "filterNum"));
            System.out.println("  卷积核高度: " + getField(convLayer, "filterHeight"));
            System.out.println("  卷积核宽度: " + getField(convLayer, "filterWidth"));
            System.out.println("  步长: " + getField(convLayer, "stride"));
            System.out.println("  填充: " + getField(convLayer, "padding"));
            System.out.println("  使用偏置: " + getField(convLayer, "useBias"));
            
        } catch (Exception e) {
            System.out.println("  ✗ 测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void testConvLayerWithBias() {
        try {
            // 创建输入形状 [1, 2, 3, 3]
            Shape inputShape = new Shape(1, 2, 3, 3);
            
            // 创建带偏置的卷积层：2个卷积核，2x2大小，步长1，填充1
            ConvLayer convLayer = new ConvLayer("bias_conv", inputShape, 2, 2, 2, 1, 1, true);
            
            System.out.println("  ✓ 带偏置卷积层创建成功");
            System.out.println("  输入形状: " + inputShape.toString());
            System.out.println("  输出形状: " + convLayer.getOutputShape().toString());
            System.out.println("  使用偏置: " + getField(convLayer, "useBias"));
            
            // 检查参数
            if (convLayer.getParamBy("filterParam") != null) {
                System.out.println("  ✓ 权重参数存在");
            } else {
                System.out.println("  ✗ 权重参数不存在");
            }
            
            if (convLayer.getParamBy("biasParam") != null) {
                System.out.println("  ✓ 偏置参数存在");
            } else {
                System.out.println("  ✗ 偏置参数不存在");
            }
            
        } catch (Exception e) {
            System.out.println("  ✗ 测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void testDifferentConvParams() {
        try {
            // 测试不同的卷积参数组合
            Shape[] inputShapes = {
                new Shape(2, 3, 32, 32),   // 2个样本，3通道，32x32
                new Shape(1, 1, 28, 28),   // 1个样本，1通道，28x28
                new Shape(4, 3, 64, 64)    // 4个样本，3通道，64x64
            };
            
            int[][] convParams = {
                {6, 5, 5, 1, 2},   // 6个5x5卷积核，步长1，填充2
                {16, 3, 3, 2, 1},  // 16个3x3卷积核，步长2，填充1
                {32, 7, 7, 1, 3}   // 32个7x7卷积核，步长1，填充3
            };
            
            for (int i = 0; i < inputShapes.length; i++) {
                Shape inputShape = inputShapes[i];
                int[] params = convParams[i];
                
                ConvLayer convLayer = new ConvLayer(
                    "test_conv_" + i, 
                    inputShape, 
                    params[0], // filterNum
                    params[1], // filterHeight
                    params[2], // filterWidth
                    params[3], // stride
                    params[4]  // padding
                );
                
                System.out.println("  ✓ 参数组合" + (i+1) + "测试通过");
                System.out.println("    输入: " + inputShape.toString());
                System.out.println("    输出: " + convLayer.getOutputShape().toString());
            }
            
        } catch (Exception e) {
            System.out.println("  ✗ 测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void testForwardPropagation() {
        try {
            // 创建简单的测试输入 [1, 1, 3, 3]
            Shape inputShape = new Shape(1, 1, 3, 3);
            ConvLayer convLayer = new ConvLayer("test_conv", inputShape, 1, 2, 2, 1, 0);
            
            // 创建全1的输入数据
            NdArray inputData = NdArray.ones(inputShape);
            Variable input = new Variable(inputData);
            
            // 执行前向传播
            Variable output = convLayer.layerForward(input);
            
            System.out.println("  ✓ 前向传播成功");
            System.out.println("    输入形状: " + input.getValue().shape.toString());
            System.out.println("    输出形状: " + output.getValue().shape.toString());
            
            // 验证输出形状
            Shape expectedOutputShape = new Shape(1, 1, 2, 2);
            if (output.getValue().shape.toString().equals(expectedOutputShape.toString())) {
                System.out.println("  ✓ 输出形状正确");
            } else {
                System.out.println("  ✗ 输出形状不正确");
                System.out.println("    期望: " + expectedOutputShape.toString());
                System.out.println("    实际: " + output.getValue().shape.toString());
            }
            
            // 测试带偏置的前向传播
            ConvLayer convLayerWithBias = new ConvLayer("test_conv_bias", inputShape, 1, 2, 2, 1, 0, true);
            Variable outputWithBias = convLayerWithBias.layerForward(input);
            
            System.out.println("  ✓ 带偏置前向传播成功");
            System.out.println("    输出形状: " + outputWithBias.getValue().shape.toString());
            
        } catch (Exception e) {
            System.out.println("  ✗ 测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    // 辅助方法：通过反射获取私有字段值（仅用于测试）
    private static Object getField(Object obj, String fieldName) {
        try {
            java.lang.reflect.Field field = obj.getClass().getDeclaredField(fieldName);
            field.setAccessible(true);
            return field.get(obj);
        } catch (Exception e) {
            return "无法获取";
        }
    }
}