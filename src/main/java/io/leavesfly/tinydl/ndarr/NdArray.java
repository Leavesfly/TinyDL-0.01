package io.leavesfly.tinydl.ndarr;

import io.leavesfly.tinydl.utils.Util;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;


/**
 * 支持更高维度的数据:1，标量;2，向量;3，矩阵;等N维度
 */
public class NdArray implements Serializable {
    
    private static final long serialVersionUID = 1L;
    /**
     * N维数组的形状
     */
    public Shape shape;

    /**
     * 真实存储数据，使用float32
     */
    public float[] buffer;

    //    # =============================================================================
    //            # NdArray的创建函数
    //    # =============================================================================
    public NdArray() {
    }

    public NdArray(Number number) {
        shape = new Shape(1, 1);
        buffer = new float[1];
        buffer[0] = number.floatValue();
    }

    public NdArray(float[] data, Shape shape) {
        if (data.length != shape.size()) {
            throw new RuntimeException("Shape error!");
        }
        this.shape = shape;
        buffer = data;
    }

    /**
     * 一维数组的初始化
     *
     * @param data
     */
    public NdArray(float[] data) {
        shape = new Shape(1, data.length);
        buffer = data;
    }


    /**
     * 二维数组的初始化
     *
     * @param data
     */
    public NdArray(float[][] data) {
        shape = new Shape(data.length, data[0].length);

        this.buffer = new float[shape.size()];
        int index = 0;
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                this.buffer[index] = data[i][j];
                index++;
            }
        }
    }

    /**
     * 三维数组的初始化
     *
     * @param data
     */
    public NdArray(float[][][] data) {
        shape = new Shape(data.length, data[0].length, data[0][0].length);

        this.buffer = new float[shape.size()];
        int index = 0;
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                for (int k = 0; k < data[i][j].length; k++) {
                    this.buffer[index] = data[i][j][k];
                    index++;
                }

            }
        }
    }

    /**
     * 四维数组的初始化
     *
     * @param data
     */
    public NdArray(float[][][][] data) {
        shape = new Shape(data.length, data[0].length, data[0][0].length, data[0][0][0].length);

        this.buffer = new float[shape.size()];
        int index = 0;
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                for (int k = 0; k < data[i][j].length; k++) {
                    for (int l = 0; l < data[i][j][k].length; l++) {
                        this.buffer[index] = data[i][j][k][l];
                        index++;
                    }
                }

            }
        }
    }

    /**
     * 通过Shape来初始化
     *
     * @param _shape
     */
    public NdArray(Shape _shape) {
        this.shape = _shape;
        buffer = new float[_shape.size()];
    }

    /**
     * 创建为0的N维数组
     *
     * @param shape
     * @return
     */
    public static NdArray zeros(Shape shape) {
        NdArray ones = new NdArray(shape);
        ones.fillAll(0.0F);
        return ones;
    }

    /**
     * 创建为1的N维数组
     *
     * @param _shape
     * @return
     */
    public static NdArray ones(Shape _shape) {
        NdArray ones = new NdArray(_shape);
        ones.fillAll(1.0f);
        return ones;
    }

    /**
     * 对角线为1，其余为0的矩阵
     */
    public static NdArray eye(Shape _shape) {
        if (!_shape.isMatrix()) {
            throw new RuntimeException("not matrix!");
        }
        NdArray eye = new NdArray(_shape);
        for (int i = 0; i < _shape.getRow(); i++) {
            for (int j = 0; j < _shape.getColumn(); j++) {
                int index = i * _shape.getRow() + j;
                eye.buffer[index] = (i == j) ? 1 : 0;
            }
        }
        return eye;
    }

    /**
     * Shape的N维数组全为number的值
     *
     * @param _shape
     * @param number
     * @return
     */
    public static NdArray like(Shape _shape, Number number) {
        NdArray like = new NdArray(_shape);
        like.fillAll(number.floatValue());
        return like;
    }

    /**
     * 类似形状的N维数组
     *
     * @param number
     * @return
     */
    public NdArray like(Number number) {
        NdArray like = new NdArray(shape);
        like.fillAll(number.floatValue());
        return like;
    }

    /**
     * 标准正态分布的随机数
     */
    public static NdArray likeRandomN(Shape _shape) {
        NdArray ndArray = new NdArray(_shape);
        Random random = new Random(0);
        int length = _shape.size();
        for (int i = 0; i < length; i++) {
            ndArray.buffer[i] = (float) random.nextGaussian();
        }
        return ndArray;
    }

    public static NdArray likeRandom(float min, float max, Shape _shape) {
        Random rand = new Random(0);
        NdArray ndArray = new NdArray(_shape);

        int length = _shape.size();
        for (int i = 0; i < length; i++) {
            ndArray.buffer[i] = rand.nextFloat() * (max - min) + min;
        }
        return ndArray;
    }


    /**
     * 在最小最大值之间生成随机的num个数，按从小到大排序
     *
     * @param min
     * @param max
     * @param num
     * @return
     */
    public static NdArray linSpace(float min, float max, int num) {
        NdArray ndArray = likeRandom(min, max, new Shape(1, num));
        Arrays.sort(ndArray.buffer);
        return ndArray;
    }

    //    # =============================================================================
    //            # 1，简单的四则运算，shape必须一样
    //    # =============================================================================

    /**
     * 加法，必须shape一样
     *
     * @param other
     * @return
     */
    public NdArray add(NdArray other) {
        if (shape.equals(other.shape)) {
            NdArray ndArray = new NdArray(shape);
            int length = shape.size();

            for (int i = 0; i < length; i++) {
                ndArray.buffer[i] = other.buffer[i] + this.buffer[i];
            }
            return ndArray;
        } else {
            throw new RuntimeException("NdArray add method shape error!");
        }
    }

    /**
     * 减法，shape必须一样
     *
     * @param other
     * @return
     */
    public NdArray sub(NdArray other) {
        if (shape.equals(other.shape)) {
            NdArray ndArray = new NdArray(shape);
            for (int i = 0; i < other.buffer.length; i++) {
                ndArray.buffer[i] = buffer[i] - other.buffer[i];
            }
            return ndArray;
        } else {
            throw new RuntimeException("NdArray sub method shape error!");
        }
    }

    /**
     * 乘法，shape必须一样
     *
     * @param other
     * @return
     */
    public NdArray mul(NdArray other) {
        if (shape.equals(other.shape)) {
            NdArray ndArray = new NdArray(shape);
            for (int i = 0; i < other.buffer.length; i++) {
                ndArray.buffer[i] = buffer[i] * other.buffer[i];
            }
            return ndArray;
        } else {
            throw new RuntimeException("NdArray sub method shape error!");
        }
    }

    /**
     * 乘法-数字
     *
     * @param number
     * @return
     */
    public NdArray mulNum(Number number) {

        NdArray ndArray = new NdArray(shape);
        for (int i = 0; i < ndArray.buffer.length; i++) {
            ndArray.buffer[i] = buffer[i] * number.floatValue();
        }
        return ndArray;
    }

    /**
     * 除法-shape必须一样
     *
     * @param other
     * @return
     */
    public NdArray div(NdArray other) {
        if (shape.equals(other.shape)) {
            NdArray ndArray = new NdArray(shape);
            for (int i = 0; i < ndArray.buffer.length; i++) {
                ndArray.buffer[i] = buffer[i] / other.buffer[i];
            }
            return ndArray;
        } else {
            throw new RuntimeException("NdArray div shape error!");
        }
    }

    /**
     * 除法-数字
     *
     * @param number
     * @return
     */
    public NdArray divNum(Number number) {

        NdArray ndArray = new NdArray(shape);
        for (int i = 0; i < ndArray.buffer.length; i++) {
            ndArray.buffer[i] = buffer[i] / number.floatValue();
        }
        return ndArray;
    }

    //    # =============================================================================
    //            # 2，简单的逻辑运算，shape必须一样
    //    # =============================================================================

    /**
     * 取反操作
     *
     * @return
     */
    public NdArray neg() {
        return zeros(this.shape).sub(this);
    }

    /**
     * 绝对值
     *
     * @return
     */
    public NdArray abs() {
        NdArray ndArray = new NdArray(shape);
        for (int i = 0; i < ndArray.buffer.length; i++) {
            ndArray.buffer[i] = Math.abs(buffer[i]);
        }
        return ndArray;
    }

    /**
     * 相等
     *
     * @param other
     * @return
     */
    public NdArray eq(NdArray other) {
        if (shape.equals(other.shape)) {
            NdArray ndArray = new NdArray(shape);
            for (int i = 0; i < buffer.length; i++) {
                if (buffer[i] == other.buffer[i]) {
                    ndArray.buffer[i] = 1f;
                } else {
                    ndArray.buffer[i] = 0f;
                }
            }
            return ndArray;
        } else {
            throw new RuntimeException("NdArray eq shape error!");
        }
    }

    /**
     * 大于
     *
     * @param other
     * @return
     */
    public NdArray gt(NdArray other) {
        if (shape.equals(other.shape)) {
            NdArray ndArray = new NdArray(shape);
            for (int i = 0; i < buffer.length; i++) {
                if (buffer[i] > other.buffer[i]) {
                    ndArray.buffer[i] = 1f;
                } else {
                    ndArray.buffer[i] = 0f;
                }
            }
            return ndArray;
        } else {
            throw new RuntimeException("NdArray gt shape error!");
        }
    }

    /**
     * 小于
     *
     * @param other
     * @return
     */
    public NdArray lt(NdArray other) {
        if (shape.equals(other.shape)) {
            NdArray ndArray = new NdArray(shape);
            for (int i = 0; i < buffer.length; i++) {
                if (buffer[i] < other.buffer[i]) {
                    ndArray.buffer[i] = 1f;
                } else {
                    ndArray.buffer[i] = 0f;
                }
            }
            return ndArray;
        } else {
            throw new RuntimeException("NdArray lt shape error!");
        }
    }

    /**
     * 矩阵的比较，全元素大于才大于
     *
     * @param other
     * @return
     */
    public boolean isLar(NdArray other) {

        if (shape.equals(other.shape)) {
            for (int i = 0; i < buffer.length; i++) {
                if (buffer[i] < other.buffer[i]) {
                    return false;
                }
            }
            return true;
        } else {
            throw new RuntimeException("NdArray isLar shape error!");
        }
    }

    //    # =============================================================================
    //            # 3，基本数学函数
    //    # =============================================================================

    /**
     * n次方
     *
     * @param number
     * @return
     */
    public NdArray pow(Number number) {
        NdArray ndArray = new NdArray(shape);
        for (int i = 0; i < ndArray.buffer.length; i++) {
            ndArray.buffer[i] = (float) Math.pow(buffer[i], number.floatValue());
        }
        return ndArray;
    }

    /**
     * 平方
     *
     * @return
     */
    public NdArray square() {
        return pow(2f);
    }

    public NdArray sqrt() {
        return pow(0.5f);
    }

    /**
     * 以e为底的指数
     *
     * @return
     */
    public NdArray exp() {
        NdArray ndArray = new NdArray(shape);
        for (int i = 0; i < ndArray.buffer.length; i++) {
            ndArray.buffer[i] = (float) Math.exp(buffer[i]);
        }
        return ndArray;
    }

    /**
     * sin 函数
     *
     * @return
     */
    public NdArray sin() {
        NdArray ndArray = new NdArray(shape);
        for (int i = 0; i < ndArray.buffer.length; i++) {
            ndArray.buffer[i] = (float) Math.sin(buffer[i]);
        }
        return ndArray;
    }

    /**
     * cos函数
     *
     * @return
     */
    public NdArray cos() {
        NdArray ndArray = new NdArray(shape);
        for (int i = 0; i < ndArray.buffer.length; i++) {
            ndArray.buffer[i] = (float) Math.cos(buffer[i]);
        }
        return ndArray;
    }

    /**
     * tanh函数
     *
     * @return
     */
    public NdArray tanh() {
        NdArray ndArray = new NdArray(shape);
        for (int i = 0; i < ndArray.buffer.length; i++) {
            ndArray.buffer[i] = (float) Math.tanh(buffer[i]);
        }
        return ndArray;
    }

    /**
     * sigmoid 函数
     *
     * @return
     */
    public NdArray sigmoid() {
        return this.mulNum(0.5f).tanh().mulNum(0.5f).add(NdArray.like(this.getShape(), 0.5f));
    }

    /**
     * 以e为底的对数
     *
     * @return
     */
    public NdArray log() {
        NdArray ndArray = new NdArray(shape);
        for (int i = 0; i < ndArray.buffer.length; i++) {
            ndArray.buffer[i] = (float) Math.log(buffer[i]);
        }
        return ndArray;
    }

    /**
     * 按行累加概率为1
     *
     * @return
     */
    public NdArray softMax() {

        if (!shape.isMatrix()) {
            throw new RuntimeException("not matrix !");
        }

        NdArray ndArray = new NdArray(shape);
        NdArray ndArrayExp = this.exp();

        for (int i = 0; i < shape.getRow(); i++) {
            float sum = 0f;
            for (int j = 0; j < shape.getColumn(); j++) {
                int index = i * shape.getColumn() + j;
                sum += ndArrayExp.buffer[index];
            }
            for (int j = 0; j < shape.getColumn(); j++) {
                int index = i * shape.getColumn() + j;
                ndArray.buffer[index] = ndArrayExp.buffer[index] / sum;
            }
        }
        return ndArray;
    }


    /**
     * 按元素，取最大值
     *
     * @param number
     * @return
     */
    public NdArray maximum(Number number) {
        NdArray ndArray = new NdArray(shape);
        for (int i = 0; i < ndArray.buffer.length; i++) {
            ndArray.buffer[i] = Math.max(buffer[i], number.floatValue());
        }
        return ndArray;
    }

    /**
     * 按元素，大于number的取1，小于取0的掩码
     *
     * @param number
     * @return
     */
    public NdArray mask(Number number) {
        NdArray ndArray = new NdArray(shape);
        for (int i = 0; i < buffer.length; i++) {
            if (buffer[i] > number.floatValue()) {
                ndArray.buffer[i] = 1f;
            } else {
                ndArray.buffer[i] = 0f;
            }
        }
        return ndArray;
    }

    //    # =============================================================================
    //            # 4，张量的变形操作
    //    # =============================================================================

    /**
     * 转置操作
     *
     * @return
     */
    public NdArray transpose() {

        if (!shape.isMatrix()) {
            throw new RuntimeException("not matrix !");
        }

        NdArray ndArray = new NdArray(new Shape(shape.getColumn(), shape.getRow()));
        for (int i = 0; i < shape.getRow(); i++) {
            for (int j = 0; j < shape.getColumn(); j++) {
                ndArray.buffer[j * shape.getRow() + i] = buffer[i * shape.getColumn() + j];
            }
        }
        return ndArray;
    }

    public NdArray transpose(int... order) {
        // 验证转置的维度顺序是否包含所有维度
        if (order.length != shape.dimension.length || Arrays.stream(order).distinct().count() != shape.dimension.length) {
            throw new IllegalArgumentException("Invalid transpose dimensions order.");
        }

        // 计算新的维度和乘数
        int[] newDimensions = new int[shape.dimension.length];
        for (int i = 0; i < order.length; i++) {
            newDimensions[i] = shape.dimension[order[i]];
        }
        NdArray transposed = new NdArray(new Shape(newDimensions));

        // 为了遍历整个数组，我们需要一个用来追踪当前位置的索引数组
        int[] indices = new int[shape.dimension.length];

        int totalElements = Arrays.stream(shape.dimension).reduce(1, (a, b) -> a * b);
        for (int i = 0; i < totalElements; i++) {
            // 将一维数组索引转换为多维数组索引
            int index = i;
            for (int j = shape.dimension.length - 1; j >= 0; j--) {
                indices[j] = index / shape.multipliers[j];
                index %= shape.multipliers[j];
            }

            // 计算转置后的索引
            int[] transposedIndices = new int[order.length];
            for (int j = 0; j < order.length; j++) {
                transposedIndices[j] = indices[order[j]];
            }

            // 将数据复制到转置后的数组中
            transposed.set(this.get(indices), transposedIndices);
        }
        return transposed;
    }

    /**
     * 变形操作
     *
     * @return
     */
    public NdArray reshape(Shape _shape) {
        if (shape.size() != _shape.size()) {
            throw new RuntimeException("_shape size is error!");
        }
        NdArray ndArray = new NdArray(_shape);
        ndArray.buffer = Arrays.copyOf(buffer, shape.size());
        return ndArray;
    }

    /**
     * 打平成只有一行的矩阵
     *
     * @return
     */
    public NdArray flatten() {
        return this.reshape(new Shape(1, shape.size()));
    }

    /**
     * 矩阵的均值
     * axis=0表示 按row
     * axis=1表示 按col
     *
     * @return
     */
    public NdArray mean(int axis) {

        if (!shape.isMatrix()) {
            throw new RuntimeException("not matrix !");
        }

        if (axis == 0) {
            NdArray ndArray = new NdArray(new Shape(1, shape.getColumn()));
            for (int i = 0; i < shape.getColumn(); i++) {
                float sum = 0f;
                for (int j = 0; j < shape.getRow(); j++) {
                    sum += buffer[j * shape.getColumn() + i];
                }
                ndArray.buffer[i] = sum / shape.getRow();
            }
            return ndArray;

        } else if (axis == 1) {
            NdArray ndArray = new NdArray(new Shape(shape.getRow(), 1));
            for (int i = 0; i < shape.getRow(); i++) {
                float sum = 0f;
                for (int j = 0; j < shape.getColumn(); j++) {
                    sum += buffer[i * shape.getColumn() + j];
                }
                ndArray.buffer[i] = sum / shape.getColumn();
            }
            return ndArray;

        } else {
            throw new RuntimeException("not impl!");
        }
    }

    /**
     * 矩阵的方差
     * axis=0表示 按row
     * axis=1表示 按col
     * mean(abs(x - x.mean())**2)
     *
     * @return
     */
    public NdArray var(int axis) {

        if (!shape.isMatrix()) {
            throw new RuntimeException("not matrix !");
        }

        if (axis == 0) {
            NdArray result = this.mean(0);
            result = this.sub(result.broadcastTo(shape));
            result = result.abs().pow(2);
            result.mean(0);
            return result;

        } else if (axis == 1) {
            NdArray result = this.mean(1);
            result = this.sub(result.broadcastTo(shape));
            result = result.pow(2);
            result.mean(1);
            return result;
        } else {
            throw new RuntimeException("not impl!");
        }
    }

    /**
     * 元素累和
     *
     * @return
     */
    public NdArray sum() {
        float sum = 0f;
        for (int i = 0; i < buffer.length; i++) {
            sum += buffer[i];
        }
        return new NdArray(sum);
    }

    /**
     * axis=0表示 按row
     * axis=1表示 按col
     *
     * @return
     */
    public NdArray sum(int axis) {

        if (!shape.isMatrix()) {
            throw new RuntimeException("not matrix !");
        }

        if (axis == 0) {
            return sumTo(new Shape(1, shape.getColumn()));

        } else if (axis == 1) {
            return sumTo(new Shape(shape.getRow(), 1));

        }
        throw new RuntimeException("not impl!");
    }

    /**
     * 按Shape进行 行或者列进行压缩累加
     *
     * @param _shape
     * @return
     */
    public NdArray sumTo(Shape _shape) {

        if (!shape.isMatrix()) {
            throw new RuntimeException("not matrix !");
        }

        if (_shape.getRow() > this.shape.getRow() || _shape.getColumn() > this.shape.getColumn()) {
            throw new RuntimeException("_shape is error!");
        }
        NdArray ndArray = new NdArray(new Shape(_shape.getRow(), _shape.getColumn()));
        for (int i = 0; i < shape.getRow(); i++) {
            for (int j = 0; j < shape.getColumn(); j++) {
                ndArray.buffer[(i % _shape.getRow()) * _shape.getColumn() + j % _shape.getColumn()] += this.buffer[j + i * this.shape.getColumn()];
            }
        }
        return ndArray;
    }

    /**
     * 广播
     *
     * @param _shape
     * @return
     */
    public NdArray broadcastTo(Shape _shape) {

        if (!shape.isMatrix()) {
            throw new RuntimeException("not matrix !");
        }

        if (_shape.getRow() < this.shape.getRow() || _shape.getColumn() < this.shape.getColumn()) {
            throw new RuntimeException("_shape is error!");
        }
        NdArray ndArray = new NdArray(new Shape(_shape.getRow(), _shape.getColumn()));

        for (int i = 0; i < _shape.getRow(); i++) {
            for (int j = 0; j < _shape.getColumn(); j++) {
                ndArray.buffer[i * _shape.getColumn() + j] += this.buffer[i % this.shape.getRow() * this.shape.getColumn() + j % this.shape.getColumn()];
            }
        }
        return ndArray;
    }

    /**
     * axis=0表示 按row
     * axis=1表示 按col
     * 返回最大值的索引
     *
     * @return
     */
    public NdArray argMax(int axis) {

        if (!shape.isMatrix()) {
            throw new RuntimeException("not matrix !");
        }

        if (axis == 0) {
            NdArray ndArray = new NdArray(new Shape(1, this.getShape().getColumn()));
            float[][] matrix = getMatrix();
            for (int i = 0; i < shape.getColumn(); i++) {
                float maxValue = Float.MIN_VALUE;
                int maxIndex = -1;
                for (int j = 0; j < shape.getRow(); j++) {
                    if (maxValue < matrix[j][i]) {
                        maxValue = matrix[j][i];
                        maxIndex = j;
                    }
                }
                ndArray.buffer[i] = maxIndex;
            }
            return ndArray;

        } else if (axis == 1) {
            NdArray ndArray = new NdArray(new Shape(this.getShape().getRow(), 1));
            float[][] matrix = getMatrix();
            for (int i = 0; i < shape.getRow(); i++) {
                float maxValue = Float.MIN_VALUE;
                int maxIndex = -1;
                for (int j = 0; j < shape.getColumn(); j++) {
                    if (maxValue < matrix[i][j]) {
                        maxValue = matrix[i][j];
                        maxIndex = j;
                    }
                }
                ndArray.buffer[i] = maxIndex;
            }
            return ndArray;
        }
        throw new RuntimeException("not impl!");
    }

    /**
     * 矩阵的内积运算，最常用的操作
     */
    public NdArray dot(NdArray other) {

        if (!shape.isMatrix()) {
            throw new RuntimeException("not matrix !");
        }

        if (shape.getColumn() != other.shape.getRow()) {
            throw new RuntimeException("NdArray dot shape.column !=other.shape.row");
        }

        NdArray ndArray = new NdArray(new Shape(shape.getRow(), other.shape.getColumn()));

        for (int i = 0; i < shape.getRow(); i++) {
            for (int j = 0; j < other.shape.getColumn(); j++) {

                float sum = 0f;
                for (int k = 0; k < shape.getColumn(); k++) {
//                    sum += this.get(i, k) * other.get(k, j);
                    sum += buffer[i * shape.getColumn() + k] * other.buffer[k * other.shape.getColumn() + j];
                }

//                ndArray.set(sum, i, j);
                ndArray.buffer[i * other.shape.getColumn() + j] = sum;
            }
        }
        return ndArray;
    }

    /**
     * 获取矩阵的一部分
     *
     * @param _rowSlices
     * @param _colSlices
     * @return
     */
    public NdArray getItem(int[] _rowSlices, int[] _colSlices) {

        if (!shape.isMatrix()) {
            throw new RuntimeException("not matrix !");
        }

        if (_rowSlices != null && _colSlices != null) {
            if (_rowSlices.length != _colSlices.length) {
                throw new RuntimeException("_rowSlices.length != _colSlices.length !");
            }

            NdArray ndArray = new NdArray(new Shape(1, _colSlices.length));
            for (int i = 0; i < _colSlices.length; i++) {
                ndArray.buffer[i] = buffer[_rowSlices[i] * shape.getColumn() + _colSlices[i]];
            }
            return ndArray;
        }

        if (_colSlices == null) {
            _colSlices = Util.getSeq(shape.getColumn());
        }
        if (_rowSlices == null) {
            _rowSlices = Util.getSeq(shape.getRow());
        }

        NdArray ndArray = new NdArray(new Shape(_rowSlices.length, _colSlices.length));
        for (int i = 0; i < _rowSlices.length; i++) {
            for (int j = 0; j < _colSlices.length; j++) {
                ndArray.buffer[i * ndArray.getShape().getColumn() + j]
                        = buffer[_rowSlices[i] * shape.getColumn() + _colSlices[j]];
            }
        }
        return ndArray;
    }

    /**
     * 设置矩阵的一部分
     *
     * @param _rowSlices
     * @param _colSlices
     * @return
     */
    public NdArray setItem(int[] _rowSlices, int[] _colSlices, float[] data) {

        if (!shape.isMatrix()) {
            throw new RuntimeException("not matrix !");
        }

        if (_rowSlices != null && _colSlices != null) {
            if (_rowSlices.length != _colSlices.length) {
                throw new RuntimeException("_rowSlices.length != _colSlices.length !");
            }

            for (int i = 0; i < _colSlices.length; i++) {
                buffer[_rowSlices[i] * shape.getColumn() + _colSlices[i]] = data[i];
            }
            return this;
        }

        throw new RuntimeException("not impl !");
    }

    /**
     * axis=0 行
     * axis=1 列
     *
     * @param axis
     * @return
     */
    public NdArray max(int axis) {

        if (!shape.isMatrix()) {
            throw new RuntimeException("not matrix !");
        }

        if (axis == 1) {
            NdArray ndArray = new NdArray(new Shape(shape.getRow(), 1));
            for (int i = 0; i < shape.getRow(); i++) {
                float max = Float.MIN_VALUE;
                for (int j = 0; j < shape.getColumn(); j++) {
                    if (max < buffer[i * shape.getColumn() + j]) {
                        max = buffer[i * shape.getColumn() + j];
                    }
                }
                ndArray.buffer[i] = max;
            }
            return ndArray;
        } else if (axis == 0) {
            //todo

        }
        throw new RuntimeException("not impl!");
    }

    /**
     * axis=0 行
     * axis=1 列
     *
     * @param axis
     * @return
     */
    public NdArray min(int axis) {

        if (!shape.isMatrix()) {
            throw new RuntimeException("not matrix !");
        }

        if (axis == 1) {
            NdArray ndArray = new NdArray(new Shape(shape.getRow(), 1));
            for (int i = 0; i < shape.getRow(); i++) {
                float min = Float.MAX_VALUE;
                for (int j = 0; j < shape.getColumn(); j++) {
                    if (min > buffer[i * shape.getColumn() + j]) {
                        min = buffer[i * shape.getColumn() + j];
                    }
                }
                ndArray.buffer[i] = min;
            }
            return ndArray;
        } else if (axis == 0) {
            //todo

        }
        throw new RuntimeException("not impl!");
    }

    /**
     * 最大值
     *
     * @return
     */
    public float max() {
        float max = Float.MIN_VALUE;
        for (float value : this.buffer) {
            if (max < value) {
                max = value;
            }
        }
        return max;
    }

    /**
     * 子NdArray
     *
     * @return
     */
    public NdArray subNdArray(int startRow, int endRow, int startCol, int endCol) {

        if (!shape.isMatrix()) {
            throw new RuntimeException("not matrix !");
        }

        NdArray ndArray = new NdArray(new Shape(endRow - startRow, endCol - startCol));
        for (int i = startRow; i < endRow; i++) {
            for (int j = startCol; j < endCol; j++) {
                ndArray.buffer[ndArray.shape.getColumn() * (i - startRow) + j - startCol] = this.buffer[i * this.shape.getColumn() + j];
            }
        }
        return ndArray;
    }


    //    # =============================================================================
    //            # 5，其他的运算
    //    # =============================================================================

    /**
     * _rowSlices 和_colSlices 都不为空，在指定位置累加
     * _rowSlices 或_colSlices 为空，在为空的 列或行累加
     *
     * @param rowSlices
     * @param colSlices
     * @param other
     * @return
     */
    public NdArray addAt(int[] rowSlices, int[] colSlices, NdArray other) {

        //TODO 需要检查输入的shape是否符合预期，否则会导致空指针，_rowSlices 或_colSlices 为空时，行或列应该和原始narray的一致
        NdArray ndArray = new NdArray(Arrays.copyOf(buffer, buffer.length), shape);

        if (colSlices != null && rowSlices != null) {
            for (int i = 0; i < rowSlices.length; i++) {
                ndArray.buffer[rowSlices[i] * ndArray.shape.getColumn() + colSlices[i]] += other.buffer[i];
            }
            return ndArray;
        }

        if (colSlices == null) {
            colSlices = Util.getSeq(shape.getColumn());
        }
        if (rowSlices == null) {
            rowSlices = Util.getSeq(shape.getRow());
        }
        for (int i = 0; i < rowSlices.length; i++) {
            for (int j = 0; j < colSlices.length; j++) {
                ndArray.buffer[rowSlices[i] * ndArray.shape.getColumn() + colSlices[j]] += other.buffer[i * other.shape.getColumn() + j];
            }
        }
        return ndArray;
    }

    /**
     * 将other 累加到当前NdArray的 i，j的开始的位置
     *
     * @param i
     * @param j
     * @param other
     */
    public NdArray addTo(int i, int j, NdArray other) {

        if (!shape.isMatrix()) {
            throw new RuntimeException("not matrix !");
        }

        for (int _i = 0; _i < other.getShape().getRow(); _i++) {
            for (int _j = 0; _j < other.getShape().getColumn(); _j++) {
                buffer[this.getShape().getColumn() * (_i + i) + _j + j] += other.buffer[other.getShape().getColumn() * _i + _j];
            }
        }
        return this;
    }

    /**
     * 在最小最大中的值
     *
     * @param min
     * @param max
     * @return
     */
    public NdArray clip(float min, float max) {
        float[] result = new float[buffer.length];

        for (int i = 0; i < buffer.length; i++) {
            if (buffer[i] > max) {
                result[i] = max;
            } else {
                result[i] = Math.max(buffer[i], min);
            }
        }

        return new NdArray(result, shape);
    }

    /**
     * 用number的值进行填充
     *
     * @param number
     */
    private void fillAll(Number number) {
        int length = buffer.length;
        for (int i = 0; i < length; i++) {
            buffer[i] = number.floatValue();
        }
    }

    /**
     * 获取第一数
     *
     * @return
     */
    public Number getNumber() {
        return buffer[0];
    }

    public Shape getShape() {
        return shape;
    }

    public void setShape(Shape shape) {
        this.shape = shape;
    }

    /**
     * 转化为二维数组返回
     */
    public float[][] getMatrix() {
        if (shape.isMatrix()) {
            float[][] matrix = new float[shape.dimension[0]][shape.dimension[1]];
            int k = 0;
            for (int i = 0; i < shape.dimension[0]; i++) {
                for (int j = 0; j < shape.dimension[1]; j++) {
                    matrix[i][j] = buffer[k];
                    k++;
                }
            }
            return matrix;
        } else if (shape.dimension.length == 1) {
            float[][] matrix = new float[1][shape.dimension[0]];
            matrix[0] = buffer;
            return matrix;
        } else {
            throw new IllegalArgumentException("不支持维度大于2");
        }
    }

    /**
     * 转化为三维数组返回
     */
    public float[][][] get3dArray() {
        if (shape.dimension.length == 3) {
            float[][][] result = new float[shape.dimension[0]][shape.dimension[1]][shape.dimension[2]];
            int index = 0;
            for (int i = 0; i < shape.dimension[0]; i++) {
                for (int j = 0; j < shape.dimension[1]; j++) {
                    for (int k = 0; k < shape.dimension[2]; k++) {
                        result[i][j][k] = buffer[index];
                        index++;
                    }
                }
            }
            return result;
        } else {
            throw new IllegalArgumentException("not support!");
        }
    }

    /**
     * 转化为四维数组返回
     */
    public float[][][][] get4dArray() {
        if (shape.dimension.length == 4) {
            float[][][][] result = new float[shape.dimension[0]][shape.dimension[1]][shape.dimension[2]][shape.dimension[3]];
            int index = 0;
            for (int i = 0; i < shape.dimension[0]; i++) {
                for (int j = 0; j < shape.dimension[1]; j++) {
                    for (int k = 0; k < shape.dimension[2]; k++) {
                        for (int l = 0; k < shape.dimension[3]; l++) {
                            result[i][j][k][l] = buffer[index];
                            index++;
                        }
                    }
                }
            }
            return result;
        } else {
            throw new IllegalArgumentException("not support!");
        }
    }

    /**
     * 打印NdArray的形状和数据
     */
    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        toStringHelper(stringBuilder, 0, new int[shape.dimension.length]);
        return "NdArray{" + "shape=" + shape + ", \n" + "data=" + stringBuilder + '}';
    }

    private void toStringHelper(StringBuilder sb, int dimIndex, int[] indices) {
        if (dimIndex == shape.dimension.length) {
            // 达到最内层维度，添加数组元素的值
            sb.append(String.format("%f", get(indices)));
            return;
        }

        sb.append("[");
        for (int i = 0; i < shape.dimension[dimIndex]; i++) {
            indices[dimIndex] = i;
            toStringHelper(sb, dimIndex + 1, indices);
            if (i < shape.dimension[dimIndex] - 1) {
                sb.append(", ");
                if (dimIndex == shape.dimension.length - 2) {
                    // 在倒数第二层维度后添加换行，以更好地显示多维数组结构
                    sb.append("\n ");
                }
            }
        }
        sb.append("]");

        if (dimIndex == 0) {
            // 在最外层维度后添加换行
            sb.append("\n");
        }
    }

    @Override
    public boolean equals(Object obj) {
        return this.toString().equals(obj.toString());
    }

    /**
     * 按维度下标设置某一个值
     *
     * @param value
     * @param _dimension
     */
    public void set(float value, int... _dimension) {
        if (_dimension.length != shape.dimension.length) {
            throw new RuntimeException("dimension.length error!");
        }
        buffer[shape.getIndex(_dimension)] = value;
    }

    /**
     * 按维度下标设置某一个值
     *
     * @param _dimension
     * @return
     */
    public float get(int... _dimension) {
        if (_dimension.length != shape.dimension.length) {
            throw new RuntimeException("dimension.length error!");
        }
        return buffer[shape.getIndex(_dimension)];
    }

}
