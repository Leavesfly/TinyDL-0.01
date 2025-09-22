package io.leavesfly.tinydl.example.transformer;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.layer.dnn.LinearLayer;

/**
 * LinearLayer简单测试示例
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * 该示例演示如何使用LinearLayer（全连接层）处理不同维度的输入数据。
 * LinearLayer是神经网络中最基础的层之一，用于执行线性变换。
 */
public class SimpleTest {
    
    /**
     * 主函数，执行LinearLayer测试
     * 
     * @param args 命令行参数
     */
    public static void main(String[] args) {
        System.out.println("=== 测试LinearLayer ===");
        
        // 创建一个简单的LinearLayer
        LinearLayer layer = new LinearLayer("test", 4, 4, true);
        
        // 测试二维输入
        System.out.println("\n1. 测试二维输入");
        NdArray input2D = NdArray.likeRandom(0.0f, 1.0f, new Shape(2, 4));
        System.out.println("输入形状: " + input2D.getShape());
        System.out.println("输入是否为矩阵: " + input2D.getShape().isMatrix());
        
        try {
            Variable output2D = layer.layerForward(new Variable(input2D));
            System.out.println("输出形状: " + output2D.getValue().getShape());
            System.out.println("二维输入测试成功！");
        } catch (Exception e) {
            System.err.println("二维输入测试失败: " + e.getMessage());
        }
        
        // 测试三维输入重塑
        System.out.println("\n2. 测试三维输入重塑");
        NdArray input3D = NdArray.likeRandom(0.0f, 1.0f, new Shape(2, 3, 4));
        System.out.println("原始三维输入形状: " + input3D.getShape());
        
        // 重塑为二维
        NdArray reshaped = input3D.reshape(new Shape(6, 4));
        System.out.println("重塑后形状: " + reshaped.getShape());
        System.out.println("重塑后是否为矩阵: " + reshaped.getShape().isMatrix());
        
        try {
            Variable outputReshaped = layer.layerForward(new Variable(reshaped));
            System.out.println("输出形状: " + outputReshaped.getValue().getShape());
            
            // 重塑回三维
            NdArray result3D = outputReshaped.getValue().reshape(new Shape(2, 3, 4));
            System.out.println("最终结果形状: " + result3D.getShape());
            System.out.println("三维输入重塑测试成功！");
        } catch (Exception e) {
            System.err.println("三维输入重塑测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}