package io.leavesfly.tinydl.example;

import io.leavesfly.tinydl.mlearning.dataset.Batch;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.func.Variable;

/**
 * Batch优化功能演示示例
 * 
 * 该示例演示了Batch类的优化功能，包括：
 * 1. Variable缓存机制，避免频繁创建对象
 * 2. 迭代器功能，支持逐个访问数据对
 * 3. 索引重置功能
 * 
 * @author TinyDL
 * @version 1.0
 */
public class BatchOptimizationExample {

    public static void main(String[] args) {
        // 创建测试数据
        NdArray[] x = new NdArray[3];
        NdArray[] y = new NdArray[3];
        
        System.out.println("=== Batch优化功能演示 ===\n");
        
        // 初始化数据
        for (int i = 0; i < 3; i++) {
            x[i] = new NdArray(new float[]{i, i + 1, i + 2});
            y[i] = new NdArray(new float[]{i});
            System.out.println("数据 " + i + ": x=" + x[i] + ", y=" + y[i]);
        }
        
        // 创建Batch实例
        Batch batch = new Batch(x, y);
        System.out.println("\n批次大小: " + batch.getSize());
        
        // 演示Variable缓存功能
        System.out.println("\n=== Variable缓存功能演示 ===");
        Variable varX1 = batch.toVariableX();
        Variable varX2 = batch.toVariableX();
        System.out.println("第一次调用toVariableX(): " + varX1);
        System.out.println("第二次调用toVariableX(): " + varX2);
        System.out.println("两次调用返回相同实例: " + (varX1 == varX2));
        
        Variable varY1 = batch.toVariableY();
        Variable varY2 = batch.toVariableY();
        System.out.println("第一次调用toVariableY(): " + varY1);
        System.out.println("第二次调用toVariableY(): " + varY2);
        System.out.println("两次调用返回相同实例: " + (varY1 == varY2));
        
        // 演示迭代器功能
        System.out.println("\n=== 迭代器功能演示 ===");
        System.out.println("初始索引: " + batch.getCurrentIndex());
        System.out.println("是否有更多数据: " + batch.hasNext());
        
        // 遍历所有数据
        while (batch.hasNext()) {
            Batch.Pair<NdArray, NdArray> pair = batch.next();
            System.out.println("索引 " + batch.getCurrentIndex() + ": x=" + pair.key + ", y=" + pair.value);
        }
        
        System.out.println("遍历完成后是否有更多数据: " + batch.hasNext());
        
        // 演示索引重置功能
        System.out.println("\n=== 索引重置功能演示 ===");
        System.out.println("重置前索引: " + batch.getCurrentIndex());
        batch.resetIndex();
        System.out.println("重置后索引: " + batch.getCurrentIndex());
        System.out.println("重置后是否有更多数据: " + batch.hasNext());
        
        // 再次遍历
        System.out.println("再次遍历数据:");
        while (batch.hasNext()) {
            Batch.Pair<NdArray, NdArray> pair = batch.next();
            System.out.println("索引 " + batch.getCurrentIndex() + ": x=" + pair.key + ", y=" + pair.value);
        }
        
        // 演示缓存清除功能
        System.out.println("\n=== 缓存清除功能演示 ===");
        Variable varXBefore = batch.toVariableX();
        System.out.println("修改数据前的Variable X: " + varXBefore);
        
        // 修改数据
        NdArray[] newX = new NdArray[3];
        for (int i = 0; i < 3; i++) {
            newX[i] = new NdArray(new float[]{i + 10, i + 11, i + 12});
        }
        batch.setX(newX);
        
        Variable varXAfter = batch.toVariableX();
        System.out.println("修改数据后的Variable X: " + varXAfter);
        System.out.println("修改数据后返回新的Variable实例: " + (varXBefore != varXAfter));
        
        System.out.println("\n=== 演示完成 ===");
    }
}