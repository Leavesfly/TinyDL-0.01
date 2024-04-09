package io.leavesfly.tinydl.nnet.layer.cnn;


public class Col2ImUtil {

    /**
     * 将列格式的数据还原成多维图像数组。
     *
     * @param col      列格式的数据
     * @param imgShape 原始图像数据的形状，形式为 [N, C, H, W]。
     * @param filterH  滤波器的高度。
     * @param filterW  滤波器的宽度。
     * @param stride   步长。
     * @param pad      填充的大小。
     * @return 还原后的图像数据，形状为 [N, C, H, W]。
     */
    public static float[][][][] col2im(float[][] col, int[] imgShape, int filterH, int filterW, int stride, int pad) {
        int N = imgShape[0], C = imgShape[1], H = imgShape[2], W = imgShape[3];
        // 输出高度和宽度的计算已经考虑了步长和填充
        int outH = (H + 2 * pad - filterH) / stride + 1;
        int outW = (W + 2 * pad - filterW) / stride + 1;

        // 创建还原图像的数组，初始化时默认所有值为0
        float[][][][] img = new float[N][C][H + 2 * pad + stride - 1][W + 2 * pad + stride - 1];

        // 遍历每个样本、通道和每个输出位置
        for (int i = 0; i < N; i++) {
            for (int c = 0; c < C; c++) {
                for (int h = 0; h < outH; h++) {
                    for (int w = 0; w < outW; w++) {
                        // 定位到当前列在col中的位置
                        int colIndex = (c * outH + h) * outW + w;
                        int nOffset = i * C * outH * outW;
                        float[] kernel = col[nOffset + colIndex];

                        // 遍历滤波器的每个元素，考虑滤波器位置
                        for (int fh = 0; fh < filterH; fh++) {
                            for (int fw = 0; fw < filterW; fw++) {
                                int imgH = h * stride + fh - pad;
                                int imgW = w * stride + fw - pad;

                                if (imgH >= 0 && imgH < H && imgW >= 0 && imgW < W) {
                                    // 只有在有效范围内才累加，而且是累加而非直接赋值，因为有可能多次访问同一位置
                                    img[i][c][imgH][imgW] += kernel[fh * filterW + fw];
                                }
                            }
                        }
                    }
                }
            }
        }

        // 如果有填充，去除填充，这一步是可选的，取决于最终所需的输出形状
        if (pad > 0) {
            return cropPadding(img, pad, H, W);
        } else {
            // 没有填充的情况下直接返回
            return img;
        }
    }

    /**
     * 从有填充的图像数组中裁剪出无填充的图像数组。
     *
     * @param img 包含填充的图像数组。
     * @param pad 填充的大小。
     * @param H   原始图像高度。
     * @param W   原始图像宽度。
     * @return 裁剪后的无填充的图像数组。
     */
    private static float[][][][] cropPadding(float[][][][] img, int pad, int H, int W) {
        int N = img.length;   // 样本数量（batch size）
        int C = img[0].length; // 通道数量
        // 创建一个不包含填充的数组，用来存储最终的结果
        float[][][][] result = new float[N][C][H][W];

        // 遍历原图像的每个维度，裁剪出不含填充的部分
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int h = 0; h < H; h++) {
                    // 由于输入图像可能包含填充，需要加上偏移量 pad 来定位到实际存储区域
                    for (int w = 0; w < W; w++) {
                        // 从有填充的图像中获取对应范围内的像素值，并赋值到结果数组
                        result[n][c][h][w] = img[n][c][h + pad][w + pad];
                    }
                }
            }
        }
        return result; // 返回裁剪后的图像数据
    }
}

