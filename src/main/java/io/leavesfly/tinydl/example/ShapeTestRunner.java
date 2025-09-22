package io.leavesfly.tinydl.example;

import io.leavesfly.tinydl.ndarr.Shape;

/**
 * 运行原有的Shape测试用例以验证兼容性
 */
public class ShapeTestRunner {
    
    public static void main(String[] args) {
        System.out.println("Shape类兼容性测试");
        System.out.println("==================");
        
        try {
            // 测试构造函数
            Shape shape1D = new Shape(5);
            Shape shape2D = new Shape(3, 4);
            Shape shape3D = new Shape(2, 3, 4);
            Shape shape4D = new Shape(2, 3, 4, 5);
            
            System.out.println("构造函数测试: 通过");
            
            // 测试size方法
            assert shape1D.size() == 5;
            assert shape2D.size() == 12;
            assert shape3D.size() == 24;
            assert shape4D.size() == 120;
            
            System.out.println("size方法测试: 通过");
            
            // 测试isMatrix方法
            assert !shape1D.isMatrix();
            assert shape2D.isMatrix();
            assert !shape3D.isMatrix();
            assert !shape4D.isMatrix();
            
            System.out.println("isMatrix方法测试: 通过");
            
            // 测试矩阵维度获取
            assert shape2D.getRow() == 3;
            assert shape2D.getColumn() == 4;
            
            System.out.println("矩阵维度获取测试: 通过");
            
            // 测试getIndex方法
            assert shape1D.getIndex(0) == 0;
            assert shape1D.getIndex(2) == 2;
            assert shape1D.getIndex(4) == 4;
            
            assert shape2D.getIndex(0, 0) == 0;
            assert shape2D.getIndex(0, 1) == 1;
            assert shape2D.getIndex(1, 0) == 4;
            assert shape2D.getIndex(2, 3) == 11;
            
            assert shape3D.getIndex(0, 0, 0) == 0;
            assert shape3D.getIndex(0, 0, 1) == 1;
            assert shape3D.getIndex(0, 1, 0) == 4;
            assert shape3D.getIndex(1, 0, 0) == 12;
            assert shape3D.getIndex(1, 2, 3) == 23;
            
            System.out.println("getIndex方法测试: 通过");
            
            // 测试equals和hashCode
            Shape s1 = new Shape(3, 4);
            Shape s2 = new Shape(3, 4);
            assert s1.equals(s2);
            assert s1.hashCode() == s2.hashCode();
            
            Shape s3 = new Shape(3, 5);
            assert !s1.equals(s3);
            
            System.out.println("equals和hashCode测试: 通过");
            
            // 测试toString
            assert "[5]".equals(shape1D.toString());
            assert "[3,4]".equals(shape2D.toString());
            assert "[2,3,4]".equals(shape3D.toString());
            assert "[2,3,4,5]".equals(shape4D.toString());
            
            System.out.println("toString方法测试: 通过");
            
            System.out.println("所有兼容性测试通过！");
            
        } catch (Exception e) {
            System.err.println("测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}