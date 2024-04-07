package io.leavesfly.tinydl.nnet.layer.cnn;


/**
 * 当处理四维数组时，我们通常处理的是多个样本（例如，一批图像）其中每个样本可能包含多个通道（例如，RGB图像有三个通道）
 */
public class Im2ColUtil {

    /**
     * 对四维输入数组执行 im2col 操作。
     *
     * @param input   预期形状为 [numSamples][channels][height][width] 的四维数组。
     * @param filterH 滤波器高度。
     * @param filterW 滤波器宽度。
     * @param stride  步长。
     * @param pad     填充。
     * @return 展开后的二维数组。
     */
    public static float[][] im2col(float[][][][] input, int filterH, int filterW, int stride, int pad) {
        int numSamples = input.length; // 样本数量（batch size）
        int channels = input[0].length; // 颜色通道数量
        int height = input[0][0].length; // 输入图像的高度
        int width = input[0][0][0].length; // 输入图像的宽度

        // 计算输出的高度和宽度
        int outHeight = (height + 2 * pad - filterH) / stride + 1;
        int outWidth = (width + 2 * pad - filterW) / stride + 1;

        // 初始化输出矩阵
        float[][] output = new float[numSamples * outHeight * outWidth][channels * filterH * filterW];

        // 对于每个样本
        for (int n = 0; n < numSamples; n++) {
            // 对于输出矩阵的每个位置
            for (int h = 0; h < outHeight; h++) {
                for (int w = 0; w < outWidth; w++) {

                    // 单列的索引（避免在内部循环中重复计算）
                    int outputRow = (n * outHeight + h) * outWidth + w;

                    // 对于每个颜色通道
                    for (int c = 0; c < channels; c++) {
                        // 对于卷积核的每个位置
                        for (int fh = 0; fh < filterH; fh++) {
                            for (int fw = 0; fw < filterW; fw++) {

                                // 计算在输入图像中对应的行列
                                int imRow = h * stride + fh - pad;
                                int imCol = w * stride + fw - pad;
                                int colIndex = (c * filterH + fh) * filterW + fw;

                                // 判断是否在输入图像的边界之内，否则填充0
                                if (imRow >= 0 && imRow < height && imCol >= 0 && imCol < width) {
                                    output[outputRow][colIndex] = input[n][c][imRow][imCol];
                                } else {
                                    output[outputRow][colIndex] = 0; // Apply padding
                                }
                            }
                        }
                    }
                }
            }
        }
        return output;
    }

}

