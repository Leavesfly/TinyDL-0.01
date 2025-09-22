package io.leavesfly.tinydl.test.dataset;

import io.leavesfly.tinydl.mlearning.dataset.Batch;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.func.Variable;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Batch类的单元测试
 * 
 * @author TinyDL
 * @version 1.0
 */
public class BatchTest {

    @Test
    public void testBatchCreation() {
        // 创建测试数据
        NdArray[] x = new NdArray[3];
        NdArray[] y = new NdArray[3];
        
        for (int i = 0; i < 3; i++) {
            x[i] = new NdArray(new float[]{i, i + 1});
            y[i] = new NdArray(new float[]{i});
        }
        
        // 创建Batch实例
        Batch batch = new Batch(x, y);
        
        // 验证基本属性
        assertEquals(3, batch.getSize());
        assertArrayEquals(x, batch.getX());
        assertArrayEquals(y, batch.getY());
    }
    
    @Test
    public void testToVariableCaching() {
        // 创建测试数据
        NdArray[] x = new NdArray[2];
        NdArray[] y = new NdArray[2];
        
        for (int i = 0; i < 2; i++) {
            x[i] = new NdArray(new float[]{i, i + 1});
            y[i] = new NdArray(new float[]{i});
        }
        
        // 创建Batch实例
        Batch batch = new Batch(x, y);
        
        // 验证Variable缓存功能
        assertSame(batch.toVariableX(), batch.toVariableX());  // 应该返回相同的实例
        assertSame(batch.toVariableY(), batch.toVariableY());  // 应该返回相同的实例
    }
    
    @Test
    public void testIteratorFunctionality() {
        // 创建测试数据
        NdArray[] x = new NdArray[3];
        NdArray[] y = new NdArray[3];
        
        for (int i = 0; i < 3; i++) {
            x[i] = new NdArray(new float[]{i, i + 1});
            y[i] = new NdArray(new float[]{i});
        }
        
        // 创建Batch实例
        Batch batch = new Batch(x, y);
        
        // 验证迭代器功能
        assertTrue(batch.hasNext());
        assertEquals(0, batch.getCurrentIndex());
        
        // 遍历所有元素
        for (int i = 0; i < 3; i++) {
            assertTrue(batch.hasNext());
            Batch.Pair<NdArray, NdArray> pair = batch.next();
            assertNotNull(pair);
            assertEquals(x[i], pair.key);
            assertEquals(y[i], pair.value);
            assertEquals(i + 1, batch.getCurrentIndex());
        }
        
        // 验证遍历完成
        assertFalse(batch.hasNext());
        assertNull(batch.next());
        
        // 验证重置功能
        batch.resetIndex();
        assertTrue(batch.hasNext());
        assertEquals(0, batch.getCurrentIndex());
    }
    
    @Test
    public void testSetterClearsCache() {
        // 创建测试数据
        NdArray[] x = new NdArray[2];
        NdArray[] y = new NdArray[2];
        
        for (int i = 0; i < 2; i++) {
            x[i] = new NdArray(new float[]{i, i + 1});
            y[i] = new NdArray(new float[]{i});
        }
        
        // 创建Batch实例
        Batch batch = new Batch(x, y);
        
        // 获取缓存的Variable实例
        Variable variableX1 = batch.toVariableX();
        Variable variableY1 = batch.toVariableY();
        
        // 修改数据
        NdArray[] newX = new NdArray[2];
        newX[0] = new NdArray(new float[]{10, 11});
        newX[1] = new NdArray(new float[]{12, 13});
        batch.setX(newX);
        
        // 验证缓存已被清除
        assertNotSame(variableX1, batch.toVariableX());
        
        // 验证Y的缓存未受影响
        assertSame(variableY1, batch.toVariableY());
    }
}