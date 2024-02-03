package io.leavesfly.tinydl.ndarr;

import io.leavesfly.tinydl.utils.Util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;


/**
 * 为了支持更高维度的数据
 */
public class NdArray2 {
    /**
     * 矩阵的形状
     */
    protected Shape shape;

    /**
     * 真实存储数据，使用float32
     */
    private float[] buffer;

    //    # =============================================================================
    //            # NdArray2的创建函数
    //    # =============================================================================
    public NdArray2() {
    }

    public NdArray2(Number number) {
        shape = new Shape(1, 1);
        buffer = new float[1];
        buffer[0] = number.floatValue();
    }

    /**
     * 不指定Shape时默认作为二维的矩阵
     *
     * @param data
     */
    public NdArray2(float[] data) {
        shape = new Shape(1, data.length);
        buffer = data;
    }

    public NdArray2(float[] data, Shape shape) {
        this.shape = shape;
        buffer = data;
    }

    public NdArray2(float[][] data) {
        shape = new Shape(data.length, data[0].length);

        this.buffer = new float[data.length * data[0].length];
        int z = 0;
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                this.buffer[z] = data[i][j];
                z++;
            }
        }
    }

    public NdArray2(Shape _shape) {
        this.shape = _shape;
        int size = 1;
        for (int dim : this.shape.dimension) {
            size *= dim;
        }
        buffer = new float[size];
    }

    /**
     * 创建为0的矩阵
     *
     * @param shape
     * @return
     */
    public static NdArray2 zeros(Shape shape) {
        NdArray2 ones = new NdArray2(shape);
        ones.fillAll(0.0F);
        return ones;
    }

    /**
     * 创建为1的矩阵
     *
     * @param _shape
     * @return
     */
    public static NdArray2 ones(Shape _shape) {
        NdArray2 ones = new NdArray2(_shape);
        ones.fillAll(1.0f);
        return ones;
    }

    /**
     * 对角线为1，其余为0的矩阵
     */
    public static NdArray2 eye(Shape _shape) {
        NdArray2 eye = new NdArray2(_shape);
        for (int i = 0; i < _shape.getRow(); i++) {
            for (int j = 0; j < _shape.getColumn(); j++) {
                eye.buffer[i * _shape.getRow() + j] = (i == j) ? 1 : 0;
            }
        }
        return eye;
    }

    /**
     * Shape的矩阵全为number的值
     *
     * @param _shape
     * @param number
     * @return
     */
    public static NdArray2 like(Shape _shape, Number number) {
        NdArray2 like = new NdArray2(_shape);
        like.fillAll(number.floatValue());
        return like;
    }

    public NdArray2 like(Number number) {
        NdArray2 like = new NdArray2(shape);
        like.fillAll(number.floatValue());
        return like;
    }

    /**
     * 标准正态分布的随机数
     */
    public static NdArray2 likeRandomN(Shape _shape) {
        NdArray2 ndArray = new NdArray2(_shape);
        Random random = new Random(0);
        int length = maxSize(_shape);
        for (int i = 0; i < length; i++) {
            ndArray.buffer[i] = (float) random.nextGaussian();
        }
        return ndArray;
    }

    public static NdArray2 likeRandom(float min, float max, Shape _shape) {
        Random rand = new Random(0);
        NdArray2 ndArray = new NdArray2(_shape);

        int length = maxSize(_shape);
        for (int i = 0; i < length; i++) {
            ndArray.buffer[i] = rand.nextFloat() * (max - min) + min;
        }
        return ndArray;
    }

    private static int maxSize(Shape _shape) {
        int size = 1;
        for (int dim : _shape.dimension) {
            size *= dim;
        }
        return size;
    }

    /**
     * 在最小最大值之间生成随机的num个数，按从小到大排序
     *
     * @param min
     * @param max
     * @param num
     * @return
     */
    public static NdArray2 linSpace(float min, float max, int num) {
        NdArray2 ndArray = likeRandom(min, max, new Shape(1, num));
        Arrays.sort(ndArray.buffer);
        return ndArray;
    }

    //    # =============================================================================
    //            # 1，四则运算
    //    # =============================================================================

    /**
     * 加法，必须shape一样
     *
     * @param other
     * @return
     */
    public NdArray2 add(NdArray2 other) {
        if (shape.equals(other.shape)) {
            NdArray2 ndArray = new NdArray2(shape);
            int length = maxSize(shape);

            for (int i = 0; i < length; i++) {
                ndArray.buffer[i] = other.buffer[i] + this.buffer[i];
            }

            return ndArray;
        } else {
            throw new RuntimeException("NdArray2 add shape error!");
        }
    }

    /**
     * 减法，shape必须一样
     *
     * @param other
     * @return
     */
    public NdArray2 sub(NdArray2 other) {
        if (shape.equals(other.shape)) {
            NdArray2 ndArray = new NdArray2(shape);
            for (int i = 0; i < other.buffer.length; i++) {
                ndArray.buffer[i] = buffer[i] - other.buffer[i];
            }
            return ndArray;
        } else {
            throw new RuntimeException("NdArray2 sub shape error!");
        }
    }

    /**
     * 乘法，shape必须一样
     *
     * @param other
     * @return
     */
    public NdArray2 mul(NdArray2 other) {
        if (shape.equals(other.shape)) {
            NdArray2 ndArray = new NdArray2(shape);
            for (int i = 0; i < other.buffer.length; i++) {
                ndArray.buffer[i] = buffer[i] * other.buffer[i];
            }
            return ndArray;
        } else {
            throw new RuntimeException("NdArray2 sub shape error!");
        }
    }

    /**
     * 乘法-数字
     *
     * @param number
     * @return
     */
    public NdArray2 mulNumber(Number number) {

        NdArray2 ndArray = new NdArray2(shape);
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
    public NdArray2 div(NdArray2 other) {
        if (shape.equals(other.shape)) {
            NdArray2 ndArray = new NdArray2(shape);
            for (int i = 0; i < ndArray.buffer.length; i++) {
                ndArray.buffer[i] = buffer[i] / other.buffer[i];
            }
            return ndArray;
        } else {
            throw new RuntimeException("NdArray2 div shape error!");
        }
    }

    /**
     * 除法-数字
     *
     * @param number
     * @return
     */
    public NdArray2 divNumber(Number number) {

        NdArray2 ndArray = new NdArray2(shape);
        for (int i = 0; i < ndArray.buffer.length; i++) {
            ndArray.buffer[i] = buffer[i] / number.floatValue();
        }
        return ndArray;
    }

    /**
     * 取反操作
     *
     * @return
     */
    public NdArray2 neg() {
        return zeros(this.shape).sub(this);
    }

    /**
     * 绝对值
     *
     * @return
     */
    public NdArray2 abs() {
        NdArray2 ndArray = new NdArray2(shape);
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
    public NdArray2 eq(NdArray2 other) {
        if (shape.equals(other.shape)) {
            NdArray2 ndArray = new NdArray2(shape);
            for (int i = 0; i < buffer.length; i++) {
                if (buffer[i] == other.buffer[i]) {
                    ndArray.buffer[i] = 1f;
                } else {
                    ndArray.buffer[i] = 0f;
                }
            }
            return ndArray;
        } else {
            throw new RuntimeException("NdArray2 eq shape error!");
        }
    }

    /**
     * 大于
     *
     * @param other
     * @return
     */
    public NdArray2 gt(NdArray2 other) {
        if (shape.equals(other.shape)) {
            NdArray2 ndArray = new NdArray2(shape);
            for (int i = 0; i < buffer.length; i++) {
                if (buffer[i] > other.buffer[i]) {
                    ndArray.buffer[i] = 1f;
                } else {
                    ndArray.buffer[i] = 0f;
                }
            }
            return ndArray;
        } else {
            throw new RuntimeException("NdArray2 gt shape error!");
        }
    }

    /**
     * 小于
     *
     * @param other
     * @return
     */
    public NdArray2 lt(NdArray2 other) {
        if (shape.equals(other.shape)) {
            NdArray2 ndArray = new NdArray2(shape);
            for (int i = 0; i < buffer.length; i++) {
                if (buffer[i] < other.buffer[i]) {
                    ndArray.buffer[i] = 1f;
                } else {
                    ndArray.buffer[i] = 0f;
                }
            }
            return ndArray;
        } else {
            throw new RuntimeException("NdArray2 lt shape error!");
        }
    }

    //    # =============================================================================
    //            # 2，基本函数
    //    # =============================================================================

    /**
     * n次方
     *
     * @param number
     * @return
     */
    public NdArray2 pow(Number number) {
        NdArray2 ndArray = new NdArray2(shape);
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
    public NdArray2 square() {
        return pow(2f);
    }

    public NdArray2 sqrt() {
        return pow(0.5f);
    }

    /**
     * 以e为底的指数
     *
     * @return
     */
    public NdArray2 exp() {
        NdArray2 ndArray = new NdArray2(shape);
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
    public NdArray2 sin() {
        NdArray2 ndArray = new NdArray2(shape);
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
    public NdArray2 cos() {
        NdArray2 ndArray = new NdArray2(shape);
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
    public NdArray2 tanh() {
        NdArray2 ndArray = new NdArray2(shape);
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
    public NdArray2 sigmoid() {
        return this.mulNumber(0.5f).tanh().mulNumber(0.5f).add(NdArray2.like(this.getShape(), 0.5f));
    }

    /**
     * 以e为底的对数
     *
     * @return
     */
    public NdArray2 log() {
        NdArray2 ndArray = new NdArray2(shape);
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
    public NdArray2 softMax() {
        NdArray2 ndArray = new NdArray2(shape);
        NdArray2 ndArrayExp = this.exp();
        for (int i = 0; i < shape.getRow(); i++) {
            float sum = 0f;
            for (int j = 0; j < shape.getColumn(); j++) {
                sum += ndArrayExp.buffer[i * shape.getColumn() + j];
            }
            for (int j = 0; j < shape.getColumn(); j++) {
                ndArray.buffer[i * shape.getColumn() + j] = ndArrayExp.buffer[i * shape.getColumn() + j] / sum;
            }
        }
        return ndArray;
    }

    /**
     * 矩阵的比较，全元素大于才大于
     *
     * @param other
     * @return
     */
    public boolean isLarger(NdArray2 other) {

        for (int i = 0; i < buffer.length; i++) {
            if (buffer[i] < other.buffer[i]) {
                return false;
            }
        }
        return true;
    }

    /**
     * 按元素，取最大值
     *
     * @param number
     * @return
     */
    public NdArray2 maximum(Number number) {
        NdArray2 ndArray = new NdArray2(shape);
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
    public NdArray2 mask(Number number) {
        NdArray2 ndArray = new NdArray2(shape);
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
    //            # 3，张量的变形操作
    //    # =============================================================================

    /**
     * 转置操作
     *
     * @return
     */
    public NdArray2 transpose() {
        NdArray2 ndArray = new NdArray2(new Shape(shape.getColumn(), shape.getRow()));
        for (int i = 0; i < shape.getRow(); i++) {
            for (int j = 0; j < shape.getColumn(); j++) {
                ndArray.buffer[j * shape.getRow() + i] = buffer[i * shape.getColumn() + j];
            }
        }
        return ndArray;
    }

    /**
     * 变形操作
     *
     * @return
     */
    public NdArray2 reshape(Shape _shape) {
        if (maxSize(shape) != maxSize(_shape)) {
            throw new RuntimeException("_shape size is error!");
        }
        NdArray2 ndArray = new NdArray2(_shape);
        ndArray.buffer = Arrays.copyOf(buffer, maxSize(shape));
        return ndArray;
    }

    public NdArray2 flatten() {
        return this.reshape(new Shape(1, maxSize(shape)));
    }

    /**
     * 矩阵的均值
     * axis=0表示 按row
     * axis=1表示 按col
     *
     * @return
     */
    public NdArray2 mean(int axis) {
        if (axis == 0) {
            NdArray2 ndArray = new NdArray2(new Shape(1, shape.getColumn()));
            for (int i = 0; i < shape.getColumn(); i++) {
                float sum = 0f;
                for (int j = 0; j < shape.getRow(); j++) {
                    sum += buffer[j * shape.getColumn() + i];
                }
                ndArray.buffer[i] = sum / shape.getRow();
            }
            return ndArray;
        } else if (axis == 1) {
            NdArray2 ndArray = new NdArray2(new Shape(shape.getRow(), 1));
            for (int i = 0; i < shape.getRow(); i++) {
                float sum = 0f;
                for (int j = 0; j < shape.getColumn(); j++) {
                    sum += buffer[i * shape.getColumn() + j];
                }
                ndArray.buffer[i] = sum / shape.getColumn();
            }
            return ndArray;
        }
        throw new RuntimeException("not impl!");
    }

    /**
     * 矩阵的方差
     * axis=0表示 按row
     * axis=1表示 按col
     * mean(abs(x - x.mean())**2)
     *
     * @return
     */
    public NdArray2 var(int axis) {
        if (axis == 0) {
            NdArray2 result = this.mean(0);
            result = this.sub(result.broadcastTo(shape));
            result = result.abs().pow(2);
            result.mean(0);
            return result;
        } else if (axis == 1) {
            NdArray2 result = this.mean(1);
            result = this.sub(result.broadcastTo(shape));
            result = result.pow(2);
            result.mean(1);
            return result;
        }
        throw new RuntimeException("not impl!");
    }

    /**
     * 元素累和
     *
     * @return
     */
    public NdArray2 sum() {
        float sum = 0f;
        for (int i = 0; i < buffer.length; i++) {
            sum += buffer[i];
        }
        return new NdArray2(sum);
    }

    /**
     * axis=0表示 按row
     * *axis=1表示 按col
     *
     * @return
     */
    public NdArray2 sum(int axis) {
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
    public NdArray2 sumTo(Shape _shape) {
        if (_shape.getRow() > this.shape.getRow() || _shape.getColumn() > this.shape.getColumn()) {
            throw new RuntimeException("_shape is error!");
        }
        NdArray2 ndArray = new NdArray2(new Shape(_shape.getRow(), _shape.getColumn()));
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
    public NdArray2 broadcastTo(Shape _shape) {
        if (_shape.getRow() < this.shape.getRow() || _shape.getColumn() < this.shape.getColumn()) {
            throw new RuntimeException("_shape is error!");
        }
        NdArray2 ndArray = new NdArray2(new Shape(_shape.getRow(), _shape.getColumn()));

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
    public NdArray2 argMax(int axis) {
        if (axis == 0) {
            NdArray2 ndArray = new NdArray2(new Shape(1, this.getShape().getColumn()));
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
            NdArray2 ndArray = new NdArray2(new Shape(this.getShape().getRow(), 1));
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
     * 矩阵的内积运算
     */
    public NdArray2 dot(NdArray2 other) {
        if (shape.getColumn() != other.shape.getRow()) {
            throw new RuntimeException("NdArray2 mul shape.column !=other.shape.row");
        }
        NdArray2 ndArray = new NdArray2(new Shape(shape.getRow(), other.shape.getColumn()));
        for (int i = 0; i < shape.getRow(); i++) {
            for (int j = 0; j < other.shape.getColumn(); j++) {
                float sum = 0f;
                for (int k = 0; k < shape.getColumn(); k++) {
                    sum += buffer[i * shape.getColumn() + k] * other.buffer[k * other.shape.getColumn() + j];
                }
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
    public NdArray2 getItem(int[] _rowSlices, int[] _colSlices) {
        if (_rowSlices != null && _colSlices != null) {
            NdArray2 ndArray = new NdArray2(new Shape(1, _colSlices.length));
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

        NdArray2 ndArray = new NdArray2(new Shape(_rowSlices.length, _colSlices.length));
        for (int i = 0; i < _rowSlices.length; i++) {
            for (int j = 0; j < _colSlices.length; j++) {
                ndArray.buffer[i * ndArray.getShape().getColumn() + j] = buffer[_rowSlices[i] * shape.getColumn() + _colSlices[j]];
            }
        }
        return ndArray;
    }

    /**
     * axis=0 行
     * axis=1 列
     *
     * @param axis
     * @return
     */
    public NdArray2 max(int axis) {
        if (axis == 1) {
            NdArray2 ndArray = new NdArray2(new Shape(shape.getRow(), 1));
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
    public NdArray2 min(int axis) {
        if (axis == 1) {
            NdArray2 ndArray = new NdArray2(new Shape(shape.getRow(), 1));
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
     * 子NdArray2
     *
     * @return
     */
    public NdArray2 subNdArray2(int startRow, int endRow, int startCol, int endCol) {
        NdArray2 ndArray = new NdArray2(new Shape(endRow - startRow, endCol - startCol));
        for (int i = startRow; i < endRow; i++) {
            for (int j = startCol; j < endCol; j++) {
                ndArray.buffer[ndArray.shape.getColumn() * (i - startRow) + j - startCol] = this.buffer[i * this.shape.getColumn() + j];
            }
        }
        return ndArray;
    }

    /**
     * 按照指定维度对多个矩阵进行叠加
     * axis=0 按行
     * axis=1 按列
     *
     * @param ndArrays
     * @return
     */
    public static NdArray2 merge(int axis, NdArray2... ndArrays) {
        NdArray2 one = ndArrays[0];
        int[] dimension = Arrays.copyOf(one.getShape().dimension, one.getShape().dimension.length);

        if (axis == 1) {
            dimension[dimension.length - 1] = dimension[dimension.length - 1] * ndArrays.length;
            NdArray2 ndArray = new NdArray2(new Shape(dimension));

            int level = maxSize(one.getShape()) / one.getShape().dimension[one.getShape().dimension.length - 1];
            int z = 0;
            for (int i = 0; i < level; i++) {
                for (NdArray2 array : ndArrays) {
                    int horizontal = array.shape.dimension[array.shape.dimension.length - 1];
                    for (int j = 0; j < horizontal; j++) {
                        ndArray.buffer[z] = array.buffer[j + i * horizontal];
                        z++;
                    }
                }
            }
            return ndArray;
        } else if (axis == 0) {
            dimension[0] = dimension[0] * ndArrays.length;
            NdArray2 ndArray = new NdArray2(new Shape(dimension));

            int z = 0;
            for (NdArray2 array : ndArrays) {
                for (int j = 0; j < array.buffer.length; j++) {
                    ndArray.buffer[z] = array.buffer[j];
                    z++;
                }
            }
            return ndArray;
        }
        throw new RuntimeException("not impl!");
    }

    //    # =============================================================================
    //            # 4，其他的运算
    //    # =============================================================================

    /**
     * _rowSlices 和_colSlices 都不为空，在指定位置累加
     * _rowSlices 或_colSlices 为空，在为空的 列或行累加
     *
     * @param _rowSlices
     * @param _colSlices
     * @param other
     * @return
     */
    public NdArray2 addAt(int[] _rowSlices, int[] _colSlices, NdArray2 other) {
        //TODO 需要检查输入的shape是否符合预期，否则会导致空指针，_rowSlices 或_colSlices 为空时，行或列应该和原始narray的一致
        NdArray2 ndArray = new NdArray2(Arrays.copyOf(buffer, buffer.length), shape);

        if (_colSlices != null && _rowSlices != null) {
            for (int i = 0; i < _rowSlices.length; i++) {
                ndArray.buffer[_rowSlices[i] * ndArray.shape.getColumn() + _colSlices[i]] += other.buffer[i];
            }
            return ndArray;
        }

        if (_colSlices == null) {
            _colSlices = Util.getSeq(shape.getColumn());
        }
        if (_rowSlices == null) {
            _rowSlices = Util.getSeq(shape.getRow());
        }
        for (int i = 0; i < _rowSlices.length; i++) {
            for (int j = 0; j < _colSlices.length; j++) {
                ndArray.buffer[_rowSlices[i] * ndArray.shape.getColumn() + _colSlices[j]] += other.buffer[i * other.shape.getColumn() + j];
            }
        }
        return ndArray;
    }

    /**
     * 将other 累加到当前NdArray2的 i，j的开始的位置
     *
     * @param i
     * @param j
     * @param other
     */
    public void addTo(int i, int j, NdArray2 other) {
        for (int _i = 0; _i < other.getShape().getRow(); _i++) {
            for (int _j = 0; _j < other.getShape().getColumn(); _j++) {
                buffer[this.getShape().getColumn() * (_i + i) + _j + j] += other.buffer[other.getShape().getColumn() * _i + _j];
            }
        }
    }

    /**
     * 在最小最大中的值
     *
     * @param min
     * @param max
     * @return
     */
    public NdArray2 clip(float min, float max) {
        float[] result = new float[buffer.length];

        for (int i = 0; i < buffer.length; i++) {
            if (buffer[i] > max) {
                result[i] = max;
            } else {
                result[i] = Math.max(buffer[i], min);
            }
        }

        return new NdArray2(result, shape);
    }

    public void fillAll(Number number) {
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
     *
     * @return
     */
    public float[][] getMatrix() {
        if (shape.dimension.length == 2) {
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
     * 表示NdArray2的形状
     */
    @Override
    public String toString() {

        List<String> temp = new ArrayList<>();
        List<String> temp2 = new ArrayList<>();
        // 将buffer转化为List
        for (float v : buffer) {
            temp2.add(String.valueOf(v));
        }

        for (int i = shape.dimension.length - 1; i > 0; i--) {

            for (int j = 0; j < temp2.size(); ) {
                StringBuilder stringBuilder = new StringBuilder();
                stringBuilder.append("[");
                for (int k = 0; k < shape.dimension[i]; k++, j++) {
                    if (k != 0) {
                        stringBuilder.append(",");
                    }
                    stringBuilder.append(temp2.get(j));
                }
                stringBuilder.append("]");
                temp.add(stringBuilder.toString());
            }

            temp2.clear();
            temp2.addAll(temp);
            temp.clear();
        }

        return "NdArray2{" + "shape=" + shape + ", buffer=" + temp2 + '}';
    }

    @Override
    public boolean equals(Object obj) {
        return this.toString().equals(obj.toString());
    }
}
