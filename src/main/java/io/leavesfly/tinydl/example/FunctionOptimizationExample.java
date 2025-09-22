package io.leavesfly.tinydl.example;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.utils.Config;

import java.util.Arrays;
import java.util.List;

/**
 * Function类优化示例
 * 
 * 展示优化后的Function类的功能和使用方法
 */
public class FunctionOptimizationExample {
    
    // 模拟一个简单的加法函数用于测试
    private static class TestAddFunction extends Function {
        @Override
        public NdArray forward(NdArray... inputs) {
            return inputs[0].add(inputs[1]);
        }
        
        @Override
        public List<NdArray> backward(NdArray yGrad) {
            return Arrays.asList(yGrad, yGrad);
        }
        
        @Override
        public int requireInputNum() {
            return 2;
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Function类优化示例");
        System.out.println("==================");
        
        // 测试1: 基本功能调用
        System.out.println("测试1: 基本功能调用");
        testBasicFunctionCall();
        
        // 测试2: 计算图构建
        System.out.println("\n测试2: 计算图构建");
        testComputationGraph();
        
        // 测试3: unChain功能
        System.out.println("\n测试3: unChain功能");
        testUnChain();
        
        // 测试4: 非训练模式
        System.out.println("\n测试4: 非训练模式");
        testNonTrainMode();
        
        // 测试5: 输入验证
        System.out.println("\n测试5: 输入验证");
        testInputValidation();
    }
    
    private static void testBasicFunctionCall() {
        TestAddFunction func = new TestAddFunction();
        Variable a = new Variable(new NdArray(3.0f));
        Variable b = new Variable(new NdArray(4.0f));
        
        Variable result = func.call(a, b);
        
        // 验证结果正确
        float expected = 7.0f;
        float actual = result.getValue().getNumber().floatValue();
        System.out.println("预期结果: " + expected);
        System.out.println("实际结果: " + actual);
        System.out.println("测试通过: " + (Math.abs(expected - actual) < 1e-6));
    }
    
    private static void testComputationGraph() {
        Config.train = true;
        
        TestAddFunction func = new TestAddFunction();
        Variable a = new Variable(new NdArray(3.0f));
        Variable b = new Variable(new NdArray(4.0f));
        
        Variable result = func.call(a, b);
        
        // 验证计算图构建
        boolean hasCreator = result.getCreator() != null;
        boolean correctCreator = result.getCreator() == func;
        boolean outputSet = func.getOutput() == result;
        boolean inputsSet = func.getInputs() != null && func.getInputs().length == 2;
        
        System.out.println("计算图构建测试:");
        System.out.println("  - 有创建者函数: " + hasCreator);
        System.out.println("  - 创建者函数正确: " + correctCreator);
        System.out.println("  - 输出变量已设置: " + outputSet);
        System.out.println("  - 输入变量已设置: " + inputsSet);
        System.out.println("计算图测试通过: " + (hasCreator && correctCreator && outputSet && inputsSet));
    }
    
    private static void testUnChain() {
        TestAddFunction func = new TestAddFunction();
        Variable a = new Variable(new NdArray(3.0f));
        Variable b = new Variable(new NdArray(4.0f));
        
        Variable result = func.call(a, b);
        
        // 验证计算图已建立
        boolean graphBuilt = func.getInputs() != null && func.getOutput() != null;
        System.out.println("计算图建立: " + graphBuilt);
        
        // 断开计算图
        func.unChain();
        
        // 验证计算图已断开
        boolean inputsCleared = func.getInputs() == null;
        boolean outputCleared = func.getOutput() == null;
        
        System.out.println("unChain测试:");
        System.out.println("  - 输入变量已清除: " + inputsCleared);
        System.out.println("  - 输出变量已清除: " + outputCleared);
        System.out.println("unChain测试通过: " + (inputsCleared && outputCleared));
    }
    
    private static void testNonTrainMode() {
        Config.train = false;
        
        TestAddFunction func = new TestAddFunction();
        Variable a = new Variable(new NdArray(3.0f));
        Variable b = new Variable(new NdArray(4.0f));
        
        Variable result = func.call(a, b);
        
        // 验证结果正确
        float expected = 7.0f;
        float actual = result.getValue().getNumber().floatValue();
        System.out.println("预期结果: " + expected);
        System.out.println("实际结果: " + actual);
        
        // 在非训练模式下，不应该构建计算图
        boolean noCreator = result.getCreator() == null;
        boolean noInputs = func.getInputs() == null;
        boolean noOutput = func.getOutput() == null;
        
        System.out.println("非训练模式测试:");
        System.out.println("  - 无创建者函数: " + noCreator);
        System.out.println("  - 无输入变量: " + noInputs);
        System.out.println("  - 无输出变量: " + noOutput);
        System.out.println("非训练模式测试通过: " + (Math.abs(expected - actual) < 1e-6 && noCreator && noInputs && noOutput));
        
        // 恢复训练模式
        Config.train = true;
    }
    
    private static void testInputValidation() {
        TestAddFunction func = new TestAddFunction();
        Variable a = new Variable(new NdArray(3.0f));
        
        try {
            // 应该抛出异常，因为加法函数需要2个输入，但只提供了1个
            func.call(a);
            System.out.println("输入验证测试: 失败 - 应该抛出异常");
        } catch (RuntimeException e) {
            System.out.println("输入验证测试: 通过 - 正确抛出异常");
            System.out.println("异常信息: " + e.getMessage());
        }
    }
}