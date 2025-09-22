package io.leavesfly.tinydl.example;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.utils.Config;

/**
 * Variable类优化示例
 * 
 * 展示优化后的Variable类的功能和使用方法
 */
public class VariableOptimizationExample {
    
    public static void main(String[] args) {
        System.out.println("Variable类优化示例");
        System.out.println("==================");
        
        // 测试1: 基本构造函数
        System.out.println("测试1: 基本构造函数");
        testConstructors();
        
        // 测试2: 梯度操作
        System.out.println("\n测试2: 梯度操作");
        testGradientOperations();
        
        // 测试3: 基本运算
        System.out.println("\n测试3: 基本运算");
        testBasicOperations();
        
        // 测试4: 新增的激活函数
        System.out.println("\n测试4: 新增的激活函数");
        testNewActivationFunctions();
        
        // 测试5: 迭代反向传播
        System.out.println("\n测试5: 迭代反向传播");
        testIterativeBackward();
        
        // 测试6: requireGrad属性
        System.out.println("\n测试6: requireGrad属性");
        testRequireGrad();
    }
    
    private static void testConstructors() {
        // 测试NdArray构造器
        NdArray arr = new NdArray(3.14f);
        Variable v1 = new Variable(arr);
        System.out.println("NdArray构造器: " + (v1.getValue() == arr));
        
        // 测试Number构造器
        Variable v2 = new Variable(2.71f);
        System.out.println("Number构造器: " + (Math.abs(v2.getValue().getNumber().floatValue() - 2.71f) < 1e-6));
        
        // 测试带名称的构造器
        Variable v3 = new Variable(arr, "test_var");
        System.out.println("带名称构造器 - 值: " + (v3.getValue() == arr));
        System.out.println("带名称构造器 - 名称: " + "test_var".equals(v3.getName()));
        
        // 测试带名称和梯度标志的构造器
        Variable v4 = new Variable(arr, "no_grad_var", false);
        System.out.println("带梯度标志构造器 - 值: " + (v4.getValue() == arr));
        System.out.println("带梯度标志构造器 - 名称: " + "no_grad_var".equals(v4.getName()));
        System.out.println("带梯度标志构造器 - requireGrad: " + !v4.isRequireGrad());
    }
    
    private static void testGradientOperations() {
        Variable var = new Variable(new NdArray(new float[][]{{1, 2}, {3, 4}}));
        
        // 初始状态梯度为null
        System.out.println("初始梯度为null: " + (var.getGrad() == null));
        
        // 设置梯度
        NdArray grad = new NdArray(new float[][]{{0.1f, 0.2f}, {0.3f, 0.4f}});
        var.setGrad(grad);
        System.out.println("设置梯度后: " + (var.getGrad() == grad));
        
        // 清除梯度
        var.clearGrad();
        System.out.println("清除梯度后为null: " + (var.getGrad() == null));
    }
    
    private static void testBasicOperations() {
        Variable a = new Variable(new NdArray(new float[][]{{1, 2}, {3, 4}}));
        Variable b = new Variable(new NdArray(new float[][]{{2, 3}, {4, 5}}));
        
        // 测试加法
        Variable sum = a.add(b);
        float[][] expectedSum = {{3, 5}, {7, 9}};
        float[][] actualSum = sum.getValue().getMatrix();
        boolean sumCorrect = compareMatrices(expectedSum, actualSum);
        System.out.println("加法运算: " + sumCorrect);
        
        // 测试乘法
        Variable mul = a.mul(b);
        float[][] expectedMul = {{2, 6}, {12, 20}};
        float[][] actualMul = mul.getValue().getMatrix();
        boolean mulCorrect = compareMatrices(expectedMul, actualMul);
        System.out.println("乘法运算: " + mulCorrect);
        
        // 测试平方
        Variable square = a.squ();
        float[][] expectedSquare = {{1, 4}, {9, 16}};
        float[][] actualSquare = square.getValue().getMatrix();
        boolean squareCorrect = compareMatrices(expectedSquare, actualSquare);
        System.out.println("平方运算: " + squareCorrect);
    }
    
    private static void testNewActivationFunctions() {
        Variable a = new Variable(new NdArray(new float[][]{{-1, 0}, {1, 2}}));
        
        // 测试Sigmoid
        Variable sigmoid = a.sigmoid();
        System.out.println("Sigmoid运算: " + (sigmoid != null && sigmoid.getValue() != null));
        
        // 测试ReLU
        Variable relu = a.relu();
        System.out.println("ReLU运算: " + (relu != null && relu.getValue() != null));
    }
    
    private static void testIterativeBackward() {
        Config.train = true;
        
        // 创建简单的计算图: z = x^2 + y
        Variable x = new Variable(new NdArray(3.0f), "x");
        Variable y = new Variable(new NdArray(2.0f), "y");
        
        Variable x_squared = x.squ(); // x^2
        Variable z = x_squared.add(y); // z = x^2 + y
        
        // 使用迭代反向传播
        z.backwardIterative();
        
        // 验证梯度
        boolean xGradExists = x.getGrad() != null;
        boolean yGradExists = y.getGrad() != null;
        
        System.out.println("迭代反向传播 - x梯度存在: " + xGradExists);
        System.out.println("迭代反向传播 - y梯度存在: " + yGradExists);
        
        if (xGradExists && yGradExists) {
            float dx = x.getGrad().getNumber().floatValue();
            float dy = y.getGrad().getNumber().floatValue();
            System.out.println("迭代反向传播 - dz/dx = 2x = " + dx + " (期望: 6)");
            System.out.println("迭代反向传播 - dz/dy = " + dy + " (期望: 1)");
            System.out.println("迭代反向传播结果正确: " + (Math.abs(dx - 6) < 1e-6 && Math.abs(dy - 1) < 1e-6));
        }
        
        Config.train = false;
    }
    
    private static void testRequireGrad() {
        Variable var = new Variable(new NdArray(1.0f));
        
        // 默认requireGrad为true
        System.out.println("默认requireGrad: " + var.isRequireGrad());
        
        // 设置为false
        var.setRequireGrad(false);
        System.out.println("设置requireGrad为false: " + !var.isRequireGrad());
        
        // 设置为true
        var.setRequireGrad(true);
        System.out.println("设置requireGrad为true: " + var.isRequireGrad());
    }
    
    private static boolean compareMatrices(float[][] expected, float[][] actual) {
        if (expected.length != actual.length) return false;
        
        for (int i = 0; i < expected.length; i++) {
            if (expected[i].length != actual[i].length) return false;
            
            for (int j = 0; j < expected[i].length; j++) {
                if (Math.abs(expected[i][j] - actual[i][j]) > 1e-6) {
                    return false;
                }
            }
        }
        
        return true;
    }
}