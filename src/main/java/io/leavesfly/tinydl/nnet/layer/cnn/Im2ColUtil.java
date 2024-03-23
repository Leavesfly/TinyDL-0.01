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
        int numSamples = input.length;
        int channels = input[0].length;
        int height = input[0][0].length;
        int width = input[0][0][0].length;

        int outHeight = (height + 2 * pad - filterH) / stride + 1;
        int outWidth = (width + 2 * pad - filterW) / stride + 1;

        float[][] output = new float[numSamples * outHeight * outWidth][channels * filterH * filterW];

        for (int n = 0; n < numSamples; n++) {
            for (int h = 0; h < outHeight; h++) {
                for (int w = 0; w < outWidth; w++) {
                    float[] column = new float[channels * filterH * filterW];
                    for (int c = 0; c < channels; c++) {
                        for (int fh = 0; fh < filterH; fh++) {
                            for (int fw = 0; fw < filterW; fw++) {
                                int imRow = h * stride + fh - pad;
                                int imCol = w * stride + fw - pad;
                                int colIndex = (c * filterH + fh) * filterW + fw;
                                if (imRow >= 0 && imRow < height && imCol >= 0 && imCol < width) {
                                    column[colIndex] = input[n][c][imRow][imCol];
                                } else {
                                    column[colIndex] = 0; // Apply padding
                                }
                            }
                        }
                    }
                    int outputRow = (n * outHeight + h) * outWidth + w;
                    output[outputRow] = column;
                }
            }
        }

        return output;
    }

    public static void main(String[] args) {
        int numSamples = 2;
        int channels = 2;
        int height = 4;
        int width = 4;

        // Initialize a sample 4D input array
        float[][][][] input = new float[numSamples][channels][height][width];
        for (int n = 0; n < numSamples; n++) {
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        input[n][c][h][w] = h * width + w + 1;
                    }
                }
            }
        }

        // Parameters for the im2col operation
        int filterH = 2;
        int filterW = 2;
        int stride = 1;
        int pad = 0;

        // Execute im2col operation
        float[][] result = im2col(input, filterH, filterW, stride, pad);

        // Print the result
        for (float[] row : result) {
            for (float val : row) {
                System.out.printf("%4.0f", val);
            }
            System.out.println();
        }
    }
}

