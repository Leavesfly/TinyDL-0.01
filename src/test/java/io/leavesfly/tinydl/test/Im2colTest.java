package io.leavesfly.tinydl.test;

import static io.leavesfly.tinydl.nnet.layer.cnn.Im2ColUtil.im2col;

public class Im2colTest {
    public static void main(String[] args) {
        // 设定测试参数
        int numSamples = 1; // batch size
        int channels = 1; // 图像通道数
        int height = 4; // 图像高度
        int width = 4; // 图像宽度
        int filterH = 2; // 卷积核高度
        int filterW = 2; // 卷积核宽度
        int stride = 1; // 步长
        int pad = 0; // 填充

        // 创建一个示例输入图像
        float[][][][] input = new float[numSamples][channels][height][width];
        for (int n = 0; n < numSamples; ++n) {
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        input[n][c][h][w] = h * width + w + 1; // 使用顺序增加的数值填充
                    }
                }
            }
        }

        // 调用im2col方法
        float[][] output = im2col(input, filterH, filterW, stride, pad);

        // 打印输出结果
        System.out.println("im2col result (rows collapsed):");
        for (float[] row : output) {
            for (float val : row) {
                System.out.printf("%4.1f ", val); // 打印单元格的值，保持一定的格式
            }
            System.out.println();
        }
    }

}

