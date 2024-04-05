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
    public static float[][][][] col2Im(float[][] col, int[] imgShape, int filterH, int filterW, int stride, int pad) {
        int N = imgShape[0], C = imgShape[1], H = imgShape[2], W = imgShape[3];
        // 计算输出的高度和宽度
        int outH = (H + 2 * pad - filterH) / stride + 1;
        int outW = (W + 2 * pad - filterW) / stride + 1;

        // 创建还原图像的数组，考虑到填充
        float[][][][] img = new float[N][C][H + 2 * pad + stride - 1][W + 2 * pad + stride - 1];

        // 遍历每个图像、通道、输出高度和宽度
        for (int i = 0; i < N; i++) {
            for (int c = 0; c < C; c++) {
                for (int h = 0; h < outH; h++) {
                    for (int w = 0; w < outW; w++) {
                        int colIndex = (c * outH + h) * outW + w;
                        int nOffset = i * C * outH * outW;
                        float[] kernel = col[nOffset + colIndex];
                        // 遍历滤波器的每个元素
                        for (int fh = 0; fh < filterH; fh++) {
                            for (int fw = 0; fw < filterW; fw++) {
                                int imgH = h * stride + fh - pad;
                                int imgW = w * stride + fw - pad;
                                // 只在图像有效区域内累加列数据
                                if (imgH >= 0 && imgH < H && imgW >= 0 && imgW < W) {
                                    img[i][c][imgH][imgW] += kernel[fh * filterW + fw];
                                }
                            }
                        }
                    }
                }
            }
        }

        // 如果有填充，去除填充
        if (pad > 0) {
            return cropPadding(img, pad, H, W);
        } else {
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
        int N = img.length;
        int C = img[0].length;
        float[][][][] result = new float[N][C][H][W];
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        // 从有填充的图像中获取对应的像素值
                        result[n][c][h][w] = img[n][c][h + pad][w + pad];
                    }
                }
            }
        }
        return result;
    }

    public static void main(String[] args) {
        // 假设col是列状数据，shape是原始图像的形状
        int[] imageShape = {2, 3, 4, 4}; // N, C, H, W
        float[][] col = new float[2 * 3 * 3 * 3][3 * 2 * 2]; // 模拟的列状数据

        // 使用col2im_array函数
        float[][][][] img = col2Im(col, imageShape, 2, 2, 1, 0); // filterH, filterW, stride, pad

        // img是输出的图像数据，可以进行进一步处理
    }
}
