package io.leavesfly.tinydl.example;

import io.leavesfly.tinydl.ndarr.Shape;

/**
 * Shape类优化示例
 * 
 * 展示优化后的Shape类的功能和使用方法
 */
public class ShapeOptimizationExample {
    
    public static void main(String[] args) {
        System.out.println("Shape类优化示例");
        System.out.println("==================");
        
        // 测试1: 基本构造函数
        System.out.println("测试1: 基本构造函数");
        testConstructors();
        
        // 测试2: 形状类型判断
        System.out.println("\n测试2: 形状类型判断");
        testShapeTypes();
        
        // 测试3: 索引计算
        System.out.println("\n测试3: 索引计算");
        testIndexCalculation();
        
        // 测试4: 维度操作
        System.out.println("\n测试4: 维度操作");
        testDimensionOperations();
        
        // 测试5: 错误处理
        System.out.println("\n测试5: 错误处理");
        testErrorHandling();
        
        // 测试6: 性能优化
        System.out.println("\n测试6: 性能优化");
        testPerformanceOptimizations();
    }
    
    private static void testConstructors() {
        // 测试标量
        Shape scalar = new Shape();
        System.out.println("标量形状: " + scalar.toString());
        
        // 测试一维
        Shape vector = new Shape(5);
        System.out.println("一维形状: " + vector.toString());
        
        // 测试二维
        Shape matrix = new Shape(3, 4);
        System.out.println("二维形状: " + matrix.toString());
        
        // 测试多维
        Shape tensor = new Shape(2, 3, 4, 5);
        System.out.println("四维形状: " + tensor.toString());
        
        // 验证维度正确性
        System.out.println("标量维度数: " + scalar.getDimNum() + " (期望: 0)");
        System.out.println("向量维度数: " + vector.getDimNum() + " (期望: 1)");
        System.out.println("矩阵维度数: " + matrix.getDimNum() + " (期望: 2)");
        System.out.println("张量维度数: " + tensor.getDimNum() + " (期望: 4)");
    }
    
    private static void testShapeTypes() {
        Shape scalar = new Shape();
        Shape vector = new Shape(5);
        Shape matrix = new Shape(3, 4);
        Shape tensor3D = new Shape(2, 3, 4);
        Shape tensor4D = new Shape(2, 3, 4, 5);
        
        System.out.println("标量是标量: " + scalar.isScalar() + " (期望: true)");
        System.out.println("标量是向量: " + scalar.isVector() + " (期望: false)");
        System.out.println("标量是矩阵: " + scalar.isMatrix() + " (期望: false)");
        
        System.out.println("向量是标量: " + vector.isScalar() + " (期望: false)");
        System.out.println("向量是向量: " + vector.isVector() + " (期望: true)");
        System.out.println("向量是矩阵: " + vector.isMatrix() + " (期望: false)");
        
        System.out.println("矩阵是标量: " + matrix.isScalar() + " (期望: false)");
        System.out.println("矩阵是向量: " + matrix.isVector() + " (期望: false)");
        System.out.println("矩阵是矩阵: " + matrix.isMatrix() + " (期望: true)");
        
        System.out.println("3D张量是矩阵: " + tensor3D.isMatrix() + " (期望: false)");
        System.out.println("4D张量是矩阵: " + tensor4D.isMatrix() + " (期望: false)");
    }
    
    private static void testIndexCalculation() {
        // 测试一维索引
        Shape vector = new Shape(5);
        System.out.println("一维索引[0]: " + vector.getIndex(0) + " (期望: 0)");
        System.out.println("一维索引[2]: " + vector.getIndex(2) + " (期望: 2)");
        System.out.println("一维索引[4]: " + vector.getIndex(4) + " (期望: 4)");
        
        // 测试二维索引
        Shape matrix = new Shape(3, 4);
        System.out.println("二维索引[0,0]: " + matrix.getIndex(0, 0) + " (期望: 0)");
        System.out.println("二维索引[0,1]: " + matrix.getIndex(0, 1) + " (期望: 1)");
        System.out.println("二维索引[1,0]: " + matrix.getIndex(1, 0) + " (期望: 4)");
        System.out.println("二维索引[2,3]: " + matrix.getIndex(2, 3) + " (期望: 11)");
        
        // 测试三维索引
        Shape tensor = new Shape(2, 3, 4);
        System.out.println("三维索引[0,0,0]: " + tensor.getIndex(0, 0, 0) + " (期望: 0)");
        System.out.println("三维索引[0,0,1]: " + tensor.getIndex(0, 0, 1) + " (期望: 1)");
        System.out.println("三维索引[0,1,0]: " + tensor.getIndex(0, 1, 0) + " (期望: 4)");
        System.out.println("三维索引[1,0,0]: " + tensor.getIndex(1, 0, 0) + " (期望: 12)");
        System.out.println("三维索引[1,2,3]: " + tensor.getIndex(1, 2, 3) + " (期望: 23)");
    }
    
    private static void testDimensionOperations() {
        Shape tensor = new Shape(2, 3, 4, 5);
        
        System.out.println("维度数: " + tensor.getDimNum() + " (期望: 4)");
        System.out.println("第0维大小: " + tensor.getDimension(0) + " (期望: 2)");
        System.out.println("第1维大小: " + tensor.getDimension(1) + " (期望: 3)");
        System.out.println("第2维大小: " + tensor.getDimension(2) + " (期望: 4)");
        System.out.println("第3维大小: " + tensor.getDimension(3) + " (期望: 5)");
        
        // 测试矩阵的行列获取
        Shape matrix = new Shape(3, 4);
        System.out.println("矩阵行数: " + matrix.getRow() + " (期望: 3)");
        System.out.println("矩阵列数: " + matrix.getColumn() + " (期望: 4)");
    }
    
    private static void testErrorHandling() {
        try {
            // 测试负维度
            new Shape(-1, 5);
            System.out.println("负维度测试: 失败 - 应该抛出异常");
        } catch (IllegalArgumentException e) {
            System.out.println("负维度测试: 通过 - 正确抛出异常: " + e.getMessage());
        }
        
        try {
            // 测试索引越界
            Shape matrix = new Shape(3, 4);
            matrix.getIndex(3, 0); // 行越界
            System.out.println("索引越界测试: 失败 - 应该抛出异常");
        } catch (IndexOutOfBoundsException e) {
            System.out.println("索引越界测试: 通过 - 正确抛出异常: " + e.getMessage());
        }
        
        try {
            // 测试维度不匹配
            Shape matrix = new Shape(3, 4);
            matrix.getIndex(0, 1, 2); // 维度不匹配
            System.out.println("维度不匹配测试: 失败 - 应该抛出异常");
        } catch (IllegalArgumentException e) {
            System.out.println("维度不匹配测试: 通过 - 正确抛出异常: " + e.getMessage());
        }
        
        try {
            // 测试非矩阵获取行列
            Shape tensor = new Shape(2, 3, 4);
            tensor.getRow();
            System.out.println("非矩阵获取行测试: 失败 - 应该抛出异常");
        } catch (IllegalStateException e) {
            System.out.println("非矩阵获取行测试: 通过 - 正确抛出异常: " + e.getMessage());
        }
    }
    
    private static void testPerformanceOptimizations() {
        Shape matrix = new Shape(100, 100);
        
        // 测试hashCode缓存
        long start = System.nanoTime();
        int hash1 = matrix.hashCode();
        long mid = System.nanoTime();
        int hash2 = matrix.hashCode();
        long end = System.nanoTime();
        
        System.out.println("第一次hashCode计算时间: " + (mid - start) + "ns");
        System.out.println("第二次hashCode计算时间: " + (end - mid) + "ns");
        System.out.println("hashCode缓存优化: " + (hash1 == hash2 ? "通过" : "失败"));
        
        // 测试equals
        Shape matrix2 = new Shape(100, 100);
        boolean equal = matrix.equals(matrix2);
        System.out.println("equals测试: " + (equal ? "通过" : "失败"));
        
        // 测试toString
        String str = matrix.toString();
        System.out.println("toString结果: " + str + " (期望: [100,100])");
    }
}