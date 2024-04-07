package io.leavesfly.tinydl.test;

import io.leavesfly.tinydl.nnet.layer.cnn.Col2ImUtil;

public class Col2ImUtilTest {

    public static void main(String[] args) {
        // 测试参数
        int N = 1; // batch size
        int C = 1; // 通道数
        int H = 4; // 图像高度
        int W = 4; // 图像宽度
        int filterH = 3; // 滤波器高度
        int filterW = 3; // 滤波器宽度
        int stride = 1; // 步长
        int pad = 1; // 填充

        // 计算输出的高度和宽度
        int outH = (H + 2 * pad - filterH) / stride + 1;
        int outW = (W + 2 * pad - filterW) / stride + 1;

        // 输入列向量初始化（随机数据）
        float[][] col = new float[N * outH * outW][C * filterH * filterW];
        for (int i = 0; i < col.length; ++i) {
            for (int j = 0; j < col[i].length; ++j) {
                col[i][j] = (float) Math.random(); // 使用随机数填充，方便测试
            }
        }

        // 原始图像形状
        int[] imgShape = {N, C, H, W};

        // 进行col2im转换
        float[][][][] reconstructedImg = Col2ImUtil.col2im(col, imgShape, filterH, filterW, stride, pad);

        // 打印结果以检查
        for (int i = 0; i < N; ++i) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        System.out.printf("%6.3f ", reconstructedImg[i][c][h][w]);
                    }
                    System.out.println();
                }
                System.out.println();
            }
            System.out.println();
        }
    }
}

