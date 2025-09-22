package io.leavesfly.tinydl.test.loss;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.mlearning.loss.MaskedSoftmaxCELoss;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * MaskedSoftmaxCELoss测试类
 */
public class MaskedSoftmaxCELossTest {

    /**
     * 测试标准Softmax交叉熵损失（无掩码）
     */
    @Test
    public void testStandardLoss() {
        // 创建预测值和真实标签
        // 预测值：batch_size=2, num_classes=3
        NdArray predict = new NdArray(new float[][]{
            {1.0f, 2.0f, 3.0f},
            {1.0f, 2.0f, 3.0f}
        });
        
        // 真实标签：one-hot编码
        NdArray label = new NdArray(new float[][]{
            {0.0f, 0.0f, 1.0f},  // 第三个类别
            {0.0f, 1.0f, 0.0f}   // 第二个类别
        });
        
        Variable predictVar = new Variable(predict);
        Variable labelVar = new Variable(label);
        
        // 创建损失函数并计算损失
        MaskedSoftmaxCELoss lossFunc = new MaskedSoftmaxCELoss();
        Variable loss = lossFunc.loss(labelVar, predictVar);
        
        // 验证损失值存在且为正数
        float lossValue = loss.getValue().getNumber().floatValue();
        assertTrue("损失值应该为正数", lossValue > 0);
        
        System.out.println("标准损失值: " + lossValue);
    }
    
    /**
     * 测试掩码Softmax交叉熵损失
     */
    @Test
    public void testMaskedLoss() {
        // 创建预测值和真实标签
        NdArray predict = new NdArray(new float[][]{
            {1.0f, 2.0f, 3.0f},
            {1.0f, 2.0f, 3.0f}
        });
        
        NdArray label = new NdArray(new float[][]{
            {0.0f, 0.0f, 1.0f},  // 第三个类别
            {0.0f, 1.0f, 0.0f}   // 第二个类别
        });
        
        Variable predictVar = new Variable(predict);
        Variable labelVar = new Variable(label);
        
        // 创建掩码：第一个样本有效，第二个样本无效
        NdArray mask = new NdArray(new float[]{1.0f, 0.0f});
        mask = mask.reshape(new Shape(2, 1)); // 调整形状以匹配损失值
        
        // 创建损失函数并设置掩码
        MaskedSoftmaxCELoss lossFunc = new MaskedSoftmaxCELoss(mask);
        Variable loss = lossFunc.loss(labelVar, predictVar);
        
        // 验证损失值存在且为正数
        float lossValue = loss.getValue().getNumber().floatValue();
        assertTrue("损失值应该为正数", lossValue > 0);
        
        System.out.println("掩码损失值: " + lossValue);
    }
    
    /**
     * 测试序列掩码创建
     */
    @Test
    public void testCreateSequenceMask() {
        // 创建序列长度数组
        NdArray lengths = new NdArray(new float[]{2, 3, 1});
        lengths = lengths.reshape(new Shape(3, 1)); // 调整为列向量
        int maxLength = 4;
        
        // 创建序列掩码
        NdArray mask = MaskedSoftmaxCELoss.createSequenceMask(lengths, maxLength);
        
        // 验证掩码形状
        assertEquals("掩码形状应该是(3, 4)", new Shape(3, 4), mask.getShape());
        
        // 验证掩码值
        // 第一个序列长度为2，前2个位置为1，后2个位置为0
        assertEquals(1.0f, mask.get(0, 0), 1e-6);
        assertEquals(1.0f, mask.get(0, 1), 1e-6);
        assertEquals(0.0f, mask.get(0, 2), 1e-6);
        assertEquals(0.0f, mask.get(0, 3), 1e-6);
        
        // 第二个序列长度为3，前3个位置为1，后1个位置为0
        assertEquals(1.0f, mask.get(1, 0), 1e-6);
        assertEquals(1.0f, mask.get(1, 1), 1e-6);
        assertEquals(1.0f, mask.get(1, 2), 1e-6);
        assertEquals(0.0f, mask.get(1, 3), 1e-6);
        
        // 第三个序列长度为1，第1个位置为1，后3个位置为0
        assertEquals(1.0f, mask.get(2, 0), 1e-6);
        assertEquals(0.0f, mask.get(2, 1), 1e-6);
        assertEquals(0.0f, mask.get(2, 2), 1e-6);
        assertEquals(0.0f, mask.get(2, 3), 1e-6);
        
        System.out.println("序列掩码创建成功");
    }
    
    /**
     * 测试因果掩码创建
     */
    @Test
    public void testCreateCausalMask() {
        int seqLen = 4;
        
        // 创建因果掩码
        NdArray mask = MaskedSoftmaxCELoss.createCausalMask(seqLen);
        
        // 验证掩码形状
        assertEquals("掩码形状应该是(4, 4)", new Shape(4, 4), mask.getShape());
        
        // 验证掩码值（上三角矩阵包括对角线为1，其他为0）
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < seqLen; j++) {
                if (j <= i) {
                    assertEquals(1.0f, mask.get(i, j), 1e-6);
                } else {
                    assertEquals(0.0f, mask.get(i, j), 1e-6);
                }
            }
        }
        
        System.out.println("因果掩码创建成功");
    }
    
    /**
     * 测试全零掩码情况
     */
    @Test
    public void testAllZeroMask() {
        // 创建预测值和真实标签
        NdArray predict = new NdArray(new float[][]{
            {1.0f, 2.0f, 3.0f},
            {1.0f, 2.0f, 3.0f}
        });
        
        NdArray label = new NdArray(new float[][]{
            {0.0f, 0.0f, 1.0f},
            {0.0f, 1.0f, 0.0f}
        });
        
        Variable predictVar = new Variable(predict);
        Variable labelVar = new Variable(label);
        
        // 创建全零掩码
        NdArray mask = new NdArray(new float[]{0.0f, 0.0f});
        mask = mask.reshape(new Shape(2, 1)); // 调整形状以匹配损失值
        
        // 创建损失函数并设置掩码
        MaskedSoftmaxCELoss lossFunc = new MaskedSoftmaxCELoss(mask);
        Variable loss = lossFunc.loss(labelVar, predictVar);
        
        // 验证损失值为0
        float lossValue = loss.getValue().getNumber().floatValue();
        assertEquals("全零掩码时损失值应该为0", 0.0f, lossValue, 1e-6);
        
        System.out.println("全零掩码测试通过，损失值: " + lossValue);
    }
}