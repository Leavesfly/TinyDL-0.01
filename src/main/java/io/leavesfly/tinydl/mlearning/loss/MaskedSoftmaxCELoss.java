package io.leavesfly.tinydl.mlearning.loss;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;

/**
 * 掩码Softmax交叉熵损失函数
 * 
 * 用于处理序列模型中的掩码交叉熵损失计算，特别适用于处理变长序列。
 * 在序列处理中，较短的序列会被填充到固定长度，掩码用于忽略填充部分的损失计算。
 * 
 * @author TinyDL
 * @version 1.0
 */
public class MaskedSoftmaxCELoss extends SoftmaxCrossEntropy {
    
    private NdArray mask;
    
    /**
     * 构造函数
     */
    public MaskedSoftmaxCELoss() {
        super();
    }
    
    /**
     * 构造函数，指定掩码
     * @param mask 掩码数组，1表示有效位置，0表示填充位置
     */
    public MaskedSoftmaxCELoss(NdArray mask) {
        super();
        this.mask = mask;
    }
    
    /**
     * 计算掩码Softmax交叉熵损失
     * 
     * @param y 真实标签
     * @param predict 预测值
     * @return 损失值变量
     */
    @Override
    public Variable loss(Variable y, Variable predict) {
        // 计算标准的Softmax交叉熵损失
        Variable standardLoss = super.loss(y, predict);
        
        // 如果提供了掩码，则应用掩码
        if (mask != null) {
            NdArray lossValue = standardLoss.getValue();
            
            // 确保掩码形状与损失值形状兼容
            NdArray broadcastMask = mask;
            if (!mask.getShape().equals(lossValue.getShape())) {
                // 如果掩码是一维的，需要广播到损失值的形状
                if (mask.getShape().dimension.length == 1) {
                    // 创建与损失值形状相同的掩码
                    broadcastMask = mask.reshape(new Shape(mask.getShape().dimension[0], 1));
                }
                // 如果需要，进一步广播到完整形状
                if (!broadcastMask.getShape().equals(lossValue.getShape())) {
                    // 检查是否可以广播
                    Shape maskShape = broadcastMask.getShape();
                    Shape lossShape = lossValue.getShape();
                    
                    // 确保目标形状的每个维度都大于等于原数组对应维度
                    if (lossShape.dimension.length >= maskShape.dimension.length) {
                        // 检查是否可以安全广播
                        boolean canBroadcast = true;
                        for (int i = 0; i < maskShape.dimension.length; i++) {
                            int maskDim = maskShape.dimension[maskShape.dimension.length - 1 - i];
                            int lossDim = lossShape.dimension[lossShape.dimension.length - 1 - i];
                            if (maskDim != 1 && maskDim != lossDim) {
                                canBroadcast = false;
                                break;
                            }
                        }
                        
                        if (canBroadcast) {
                            broadcastMask = broadcastMask.broadcastTo(lossShape);
                        } else {
                            // 如果无法直接广播，创建一个新的掩码
                            broadcastMask = new NdArray(lossShape);
                            // 将原掩码的值复制到新掩码中（简化处理）
                            for (int i = 0; i < Math.min(broadcastMask.getShape().size(), maskShape.size()); i++) {
                                broadcastMask.buffer[i] = mask.buffer[i % mask.buffer.length];
                            }
                        }
                    } else {
                        // 如果无法直接广播，创建一个新的掩码
                        broadcastMask = new NdArray(lossShape);
                        // 将原掩码的值复制到新掩码中（简化处理）
                        for (int i = 0; i < Math.min(broadcastMask.getShape().size(), maskShape.size()); i++) {
                            broadcastMask.buffer[i] = mask.buffer[i % mask.buffer.length];
                        }
                    }
                }
            }
            
            // 应用掩码：将填充位置的损失设为0
            NdArray maskedLoss = lossValue.mul(broadcastMask);
            
            // 计算平均损失时只考虑有效位置
            float validPositions = broadcastMask.sum().getNumber().floatValue();
            if (validPositions > 0) {
                // 返回掩码后的平均损失
                return new Variable(maskedLoss.sum().divNum(validPositions));
            } else {
                // 如果没有有效位置，返回0损失
                return new Variable(0f);
            }
        }
        
        // 如果没有提供掩码，返回标准损失
        return standardLoss;
    }
    
    /**
     * 设置掩码
     * @param mask 掩码数组，1表示有效位置，0表示填充位置
     */
    public void setMask(NdArray mask) {
        this.mask = mask;
    }
    
    /**
     * 获取掩码
     * @return 掩码数组
     */
    public NdArray getMask() {
        return mask;
    }
    
    /**
     * 创建序列掩码
     * 
     * @param lengths 序列长度数组，形状为(batch_size,)
     * @param maxLength 最大序列长度
     * @return 掩码数组，形状为(batch_size, max_length)
     */
    public static NdArray createSequenceMask(NdArray lengths, int maxLength) {
        int batchSize = lengths.getShape().dimension[0];
        
        NdArray mask = new NdArray(new Shape(batchSize, maxLength));
        
        // 为每个批次创建掩码
        for (int b = 0; b < batchSize; b++) {
            // 正确获取序列长度值
            int length = (int) lengths.buffer[b];
            // 将有效位置设为1
            for (int i = 0; i < Math.min(length, maxLength); i++) {
                mask.set(1.0f, b, i);
            }
            // 填充位置保持为0
        }
        
        return mask;
    }
    
    /**
     * 创建因果掩码（用于解码器自回归预测）
     * 
     * @param seqLen 序列长度
     * @return 因果掩码数组，形状为(seq_len, seq_len)
     */
    public static NdArray createCausalMask(int seqLen) {
        NdArray mask = new NdArray(new Shape(seqLen, seqLen));
        
        // 上三角矩阵（包括对角线）设为1，其他位置为0
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j <= i; j++) {
                mask.set(1.0f, i, j);
            }
        }
        
        return mask;
    }
}