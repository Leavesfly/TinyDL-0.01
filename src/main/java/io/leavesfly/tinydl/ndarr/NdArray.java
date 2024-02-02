package io.leavesfly.tinydl.ndarr;

import io.leavesfly.tinydl.utils.Util;

import java.util.Arrays;
import java.util.Random;

/**
 * 支持，1，标量;2，向量;3，矩阵，
 * <p>
 * 暂不支持，更高纬度的张量，更高维的通过Tensor来继承简单实现
 */
public class NdArray {
    /**
     * 矩阵的形状
     */
    protected Shape shape;

    /**
     * 真实存储数据，使用float32
     */
    private float[][] matrix;

    //    # =============================================================================
//            # NdArray的创建函数
//    # =============================================================================
    public NdArray() {
    }

    public NdArray(Number number) {
        shape = new Shape(1, 1);
        matrix = new float[1][1];
        matrix[0][0] = number.floatValue();
    }

    public NdArray(float[] data) {
        shape = new Shape(1, data.length);
        matrix = new float[shape.getRow()][shape.getColumn()];
        matrix[0] = data;
    }

    public NdArray(float[][] data) {
        shape = new Shape(data.length, data[0].length);
        matrix = data;
    }

    public NdArray(Shape _shape) {
        this.shape = new Shape(_shape.getRow(), _shape.getColumn());
        matrix = new float[_shape.getRow()][_shape.getColumn()];
    }

    /**
     * 创建为0的矩阵
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
     * 创建为1的矩阵
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
        NdArray eye = new NdArray(_shape);
        for (int i = 0; i < _shape.getRow(); i++) {
            for (int j = 0; j < _shape.getColumn(); j++) {
                eye.getMatrix()[i][j] = (i == j) ? 1 : 0;
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
    public static NdArray like(Shape _shape, Number number) {
        NdArray like = new NdArray(_shape);
        like.fillAll(number.floatValue());
        return like;
    }

    public NdArray like(Number number) {
        NdArray like = new NdArray(shape);
        like.fillAll(number.floatValue());
        return like;
    }

    /**
     * 标准正态分布的随机数
     */
    public static NdArray likeRandomN(Shape _shape) {
        NdArray like = new NdArray(_shape);
        Random random = new Random(0);
        for (int i = 0; i < _shape.getRow(); i++) {
            for (int j = 0; j < _shape.getColumn(); j++) {
                like.matrix[i][j] = (float) random.nextGaussian();
            }
        }
        return like;
    }

    public static NdArray likeRandom(float min, float max, Shape _shape) {
        Random rand = new Random(0);
        NdArray ndArray = new NdArray(_shape);
        for (int i = 0; i < _shape.getRow(); i++) {
            for (int j = 0; j < _shape.getColumn(); j++) {
                ndArray.matrix[i][j] = rand.nextFloat() * (max - min) + min;
            }
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
        Arrays.sort(ndArray.getMatrix()[0]);
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
    public NdArray add(NdArray other) {
        if (shape.equals(other.shape)) {
            NdArray ndArray = new NdArray(shape);
            for (int i = 0; i < shape.getRow(); i++) {
                for (int j = 0; j < shape.getColumn(); j++) {
                    ndArray.getMatrix()[i][j] = matrix[i][j] + other.getMatrix()[i][j];
                }
            }
            return ndArray;
        } else {
            throw new RuntimeException("NdArray add shape error!");
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
            for (int i = 0; i < shape.getRow(); i++) {
                for (int j = 0; j < shape.getColumn(); j++) {
                    ndArray.matrix[i][j] = matrix[i][j] - other.getMatrix()[i][j];
                }
            }
            return ndArray;
        } else {
            throw new RuntimeException("NdArray sub shape error!");
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
            for (int i = 0; i < shape.getRow(); i++) {
                for (int j = 0; j < shape.getColumn(); j++) {
                    ndArray.matrix[i][j] = matrix[i][j] * other.getMatrix()[i][j];
                }
            }
            return ndArray;
        } else {
            throw new RuntimeException("NdArray sub shape error!");
        }
    }

    /**
     * 乘法-数字
     *
     * @param number
     * @return
     */
    public NdArray mulNumber(Number number) {

        NdArray ndArray = new NdArray(shape);
        for (int i = 0; i < shape.getRow(); i++) {
            for (int j = 0; j < shape.getColumn(); j++) {
                ndArray.getMatrix()[i][j] = matrix[i][j] * number.floatValue();
            }
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
            for (int i = 0; i < shape.getRow(); i++) {
                for (int j = 0; j < shape.getColumn(); j++) {
                    ndArray.getMatrix()[i][j] = matrix[i][j] / other.matrix[i][j];
                }
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
    public NdArray divNumber(Number number) {

        NdArray ndArray = new NdArray(shape);
        for (int i = 0; i < shape.getRow(); i++) {
            for (int j = 0; j < shape.getColumn(); j++) {
                ndArray.getMatrix()[i][j] = matrix[i][j] / number.floatValue();
            }
        }
        return ndArray;
    }

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
        for (int i = 0; i < shape.getRow(); i++) {
            for (int j = 0; j < shape.getColumn(); j++) {
                if (matrix[i][j] < 0) {
                    ndArray.matrix[i][j] = -matrix[i][j];
                } else {
                    ndArray.matrix[i][j] = matrix[i][j];
                }
            }
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
            for (int i = 0; i < shape.getRow(); i++) {
                for (int j = 0; j < shape.getColumn(); j++) {
                    if (matrix[i][j] == other.matrix[i][j]) {
                        ndArray.getMatrix()[i][j] = 1f;
                    } else {
                        ndArray.getMatrix()[i][j] = 0f;
                    }
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
            for (int i = 0; i < shape.getRow(); i++) {
                for (int j = 0; j < shape.getColumn(); j++) {
                    if (matrix[i][j] > other.matrix[i][j]) {
                        ndArray.getMatrix()[i][j] = 1f;
                    } else {
                        ndArray.getMatrix()[i][j] = 0f;
                    }
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
            for (int i = 0; i < shape.getRow(); i++) {
                for (int j = 0; j < shape.getColumn(); j++) {
                    if (matrix[i][j] > other.matrix[i][j]) {
                        ndArray.getMatrix()[i][j] = 1f;
                    } else {
                        ndArray.getMatrix()[i][j] = 0f;
                    }
                }
            }
            return ndArray;
        } else {
            throw new RuntimeException("NdArray lt shape error!");
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
    public NdArray pow(Number number) {
        NdArray ndArray = new NdArray(shape);
        for (int i = 0; i < shape.getRow(); i++) {
            for (int j = 0; j < shape.getColumn(); j++) {
                ndArray.getMatrix()[i][j] = (float) Math.pow(matrix[i][j], number.floatValue());
            }
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
        for (int i = 0; i < shape.getRow(); i++) {
            for (int j = 0; j < shape.getColumn(); j++) {
                ndArray.getMatrix()[i][j] = (float) Math.exp(matrix[i][j]);
            }
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
        for (int i = 0; i < shape.getRow(); i++) {
            for (int j = 0; j < shape.getColumn(); j++) {
                ndArray.getMatrix()[i][j] = (float) Math.sin(matrix[i][j]);
            }
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
        for (int i = 0; i < shape.getRow(); i++) {
            for (int j = 0; j < shape.getColumn(); j++) {
                ndArray.getMatrix()[i][j] = (float) Math.cos(matrix[i][j]);
            }
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
        for (int i = 0; i < shape.getRow(); i++) {
            for (int j = 0; j < shape.getColumn(); j++) {
                ndArray.getMatrix()[i][j] = (float) Math.tanh(matrix[i][j]);
            }
        }
        return ndArray;
    }

    /**
     * sigmoid 函数
     *
     * @return
     */
    public NdArray sigmoid() {
        return this.mulNumber(0.5f).tanh().mulNumber(0.5f).add(NdArray.like(this.getShape(), 0.5f));
    }


    /**
     * 以e为底的对数
     *
     * @return
     */
    public NdArray log() {
        NdArray ndArray = new NdArray(shape);
        for (int i = 0; i < shape.getRow(); i++) {
            for (int j = 0; j < shape.getColumn(); j++) {
                ndArray.getMatrix()[i][j] = (float) Math.log(matrix[i][j]);
            }
        }
        return ndArray;
    }

    /**
     * 按行累加概率为1
     *
     * @return
     */
    public NdArray softMax() {
        NdArray ndArray = new NdArray(shape);
        NdArray ndArrayExp = this.exp();
        for (int i = 0; i < shape.getRow(); i++) {
            float sum = 0f;
            for (int j = 0; j < shape.getColumn(); j++) {
                sum += ndArrayExp.getMatrix()[i][j];
            }
            for (int j = 0; j < shape.getColumn(); j++) {
                ndArray.getMatrix()[i][j] = ndArrayExp.getMatrix()[i][j] / sum;
            }
        }
        return ndArray;
    }

    /**
     * 矩阵的比较，全元素大于才大于
     *
     * @param Other
     * @return
     */
    public boolean isLarger(NdArray Other) {
        for (int i = 0; i < shape.getRow(); i++) {
            for (int j = 0; j < shape.getColumn(); j++) {
                if (matrix[i][j] < Other.matrix[i][j]) {
                    return false;
                }
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
    public NdArray maximum(Number number) {
        NdArray ndArray = new NdArray(shape);
        for (int i = 0; i < shape.getRow(); i++) {
            for (int j = 0; j < shape.getColumn(); j++) {
                ndArray.matrix[i][j] = Math.max(ndArray.matrix[i][j], number.floatValue());
            }
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
        for (int i = 0; i < shape.getRow(); i++) {
            for (int j = 0; j < shape.getColumn(); j++) {
                if (this.matrix[i][j] > number.floatValue()) {
                    ndArray.matrix[i][j] = 1;
                } else {
                    ndArray.matrix[i][j] = 0;
                }
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
    public NdArray transpose() {
        NdArray ndArray = new NdArray(new Shape(shape.getColumn(), shape.getRow()));
        for (int i = 0; i < shape.getRow(); i++) {
            for (int j = 0; j < shape.getColumn(); j++) {
                ndArray.getMatrix()[j][i] = matrix[i][j];
            }
        }
        return ndArray;
    }

    /**
     * 变形操作
     *
     * @return
     */
    public NdArray reshape(Shape _shape) {
        if (shape.getRow() * shape.getColumn() != _shape.getRow() * _shape.getColumn()) {
            throw new RuntimeException("_shape size is error!");
        }
        float[][] _matrix = new float[_shape.getRow()][_shape.getColumn()];
        for (int i = 0; i < _shape.getRow() * _shape.getColumn(); i++) {
            _matrix[i / _shape.getColumn()][i % _shape.getColumn()] = matrix[i / shape.getRow()][i % shape.getColumn()];
        }
        return new NdArray(_matrix);
    }

    public NdArray flatten() {
        return this.reshape(new Shape(1, shape.getRow() * shape.getColumn()));
    }


    /**
     * 矩阵的均值
     * axis=0表示 按row
     * axis=1表示 按col
     *
     * @return
     */
    public NdArray mean(int axis) {
        if (axis == 0) {
            NdArray ndArray = new NdArray(new Shape(1, shape.getColumn()));
            for (int i = 0; i < shape.getColumn(); i++) {
                float sum = 0f;
                for (int j = 0; j < shape.getRow(); j++) {
                    sum += matrix[j][i];
                }
                ndArray.getMatrix()[0][i] = sum / shape.getRow();
            }
            return ndArray;
        } else if (axis == 1) {
            NdArray ndArray = new NdArray(new Shape(shape.getRow(), 1));
            for (int i = 0; i < shape.getRow(); i++) {
                float sum = 0f;
                for (int j = 0; j < shape.getColumn(); j++) {
                    sum += matrix[i][j];
                }
                ndArray.getMatrix()[i][0] = sum / shape.getColumn();
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
    public NdArray var(int axis) {
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
        }
        throw new RuntimeException("not impl!");
    }


    /**
     * 元素累和
     *
     * @return
     */
    public NdArray sum() {
        float sum = 0f;
        for (int i = 0; i < shape.getRow(); i++) {
            for (int j = 0; j < shape.getColumn(); j++) {
                sum += matrix[i][j];
            }
        }
        return new NdArray(sum);
    }

    /**
     * axis=0表示 按row
     * *axis=1表示 按col
     *
     * @return
     */
    public NdArray sum(int axis) {
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
        if (_shape.getRow() > this.matrix.length || _shape.getColumn() > this.matrix[0].length) {
            throw new RuntimeException("_shape is error!");
        }
        float[][] _matrix = new float[_shape.getRow()][_shape.getColumn()];
        for (int i = 0; i < this.matrix.length; i++) {
            for (int j = 0; j < this.matrix[0].length; j++) {
                _matrix[i % _matrix.length][j % _matrix[0].length] += matrix[i][j];
            }
        }
        return new NdArray(_matrix);
    }

    /**
     * 广播
     *
     * @param _shape
     * @return
     */
    public NdArray broadcastTo(Shape _shape) {
        if (_shape.getRow() < this.matrix.length || _shape.getColumn() < this.matrix[0].length) {
            throw new RuntimeException("_shape is error!");
        }
        float[][] _matrix = new float[_shape.getRow()][_shape.getColumn()];
        for (int i = 0; i < _shape.getRow(); i++) {
            for (int j = 0; j < _shape.getColumn(); j++) {
                _matrix[i][j] = matrix[i % matrix.length][j % matrix[0].length];
            }
        }
        return new NdArray(_matrix);
    }

    /**
     * axis=0表示 按row
     * *axis=1表示 按col
     * 返回最大值的索引
     *
     * @return
     */
    public NdArray argMax(int axis) {
        if (axis == 0) {
            NdArray ndArray = new NdArray(new Shape(1, this.getShape().getColumn()));
            for (int i = 0; i < shape.getColumn(); i++) {
                float maxValue = Float.MIN_VALUE;
                int maxIndex = -1;
                for (int j = 0; j < shape.getRow(); j++) {
                    if (maxValue < getMatrix()[j][i]) {
                        maxValue = getMatrix()[j][i];
                        maxIndex = j;
                    }
                }
                ndArray.getMatrix()[0][i] = maxIndex;
            }
            return ndArray;
        } else if (axis == 1) {
            NdArray ndArray = new NdArray(new Shape(this.getShape().getRow(), 1));
            for (int i = 0; i < shape.getRow(); i++) {
                float maxValue = Float.MIN_VALUE;
                int maxIndex = -1;
                for (int j = 0; j < shape.getColumn(); j++) {
                    if (maxValue < getMatrix()[i][j]) {
                        maxValue = getMatrix()[i][j];
                        maxIndex = j;
                    }
                }
                ndArray.getMatrix()[i][0] = maxIndex;
            }
            return ndArray;
        }
        throw new RuntimeException("not impl!");
    }


    /**
     * 向量的内积运算
     */
    public NdArray dot(NdArray other) {
        if (shape.getColumn() != other.shape.getRow()) {
            throw new RuntimeException("NdArray mul shape.column !=other.shape.row");
        }
        NdArray ndArray = new NdArray(new Shape(shape.getRow(), other.shape.getColumn()));
        for (int i = 0; i < shape.getRow(); i++) {
            for (int j = 0; j < other.shape.getColumn(); j++) {
                float sum = 0f;
                for (int k = 0; k < shape.getColumn(); k++) {
                    sum += matrix[i][k] * other.matrix[k][j];
                }
                ndArray.matrix[i][j] = sum;
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
        if (_rowSlices != null && _colSlices != null) {
            NdArray ndArray = new NdArray(new Shape(1, _colSlices.length));
            for (int i = 0; i < _colSlices.length; i++) {
                ndArray.matrix[0][i] = matrix[_rowSlices[i]][_colSlices[i]];
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
                ndArray.matrix[i][j] = matrix[_rowSlices[i]][_colSlices[j]];
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
    public NdArray max(int axis) {
        if (axis == 1) {
            NdArray ndArray = new NdArray(new Shape(shape.getRow(), 1));
            for (int i = 0; i < this.matrix.length; i++) {
                float max = Float.MIN_VALUE;
                for (int j = 0; j < this.matrix[0].length; j++) {
                    if (max < matrix[i][j]) {
                        max = matrix[i][j];
                    }
                }
                ndArray.getMatrix()[i][0] = max;
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
    public NdArray min(int axis) {
        if (axis == 1) {
            NdArray ndArray = new NdArray(new Shape(shape.getRow(), 1));
            for (int i = 0; i < this.matrix.length; i++) {
                float min = Float.MAX_VALUE;
                for (int j = 0; j < this.matrix[0].length; j++) {
                    if (min > matrix[i][j]) {
                        min = matrix[i][j];
                    }
                }
                ndArray.getMatrix()[i][0] = min;
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
        for (float[] floats : this.matrix) {
            for (int j = 0; j < this.matrix[0].length; j++) {
                if (max < floats[j]) {
                    max = floats[j];
                }
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
        NdArray ndArray = new NdArray(new Shape(endRow - startRow, endCol - startCol));
        for (int i = startRow; i < endRow; i++) {
            for (int j = startCol; j < endCol; j++) {
                ndArray.getMatrix()[i - startRow][j - startCol] = matrix[i][j];
            }
        }
        return ndArray;
    }

    /**
     * 按行或列进行叠加
     * axis=0 按 行
     * axis=1 按列(暂没实现)
     *
     * @param ndArrays
     * @return
     */
    public static NdArray merge(int axis, NdArray... ndArrays) {
        if (axis == 0) {
            NdArray one = ndArrays[0];
            if (one.getShape().getRow() != 1) {
                throw new RuntimeException(" NdArray merge（） ndArrays  error！");
            }
            int size = ndArrays.length;
            int column = one.getShape().getColumn();
            NdArray ndArray = new NdArray(new Shape(size, column));
            for (int i = 0; i < size; i++) {
                ndArray.matrix[i] = ndArrays[i].matrix[0];
            }
            return ndArray;
        } else if (axis == 1) {
            NdArray one = ndArrays[0];
            if (one.getShape().getColumn() != 1) {
                throw new RuntimeException(" NdArray merge ndArrays error！");
            }
            int row = one.getShape().getRow();
            int size = ndArrays.length;
            NdArray ndArray = new NdArray(new Shape(row, size));
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < size; j++) {
                    ndArray.matrix[i][j] = ndArrays[j].getMatrix()[i][0];
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
    public NdArray addAt(int[] _rowSlices, int[] _colSlices, NdArray other) {
        NdArray ndArray = new NdArray(shape);
        for (int i = 0; i < shape.getRow(); i++) {
            ndArray.getMatrix()[i] = Arrays.copyOfRange(matrix[i], 0, matrix.length);
        }

        if (_colSlices != null && _rowSlices != null) {
            for (int i = 0; i < _rowSlices.length; i++) {
                ndArray.matrix[_rowSlices[i]][_colSlices[i]] += other.matrix[0][i];
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
                ndArray.matrix[_rowSlices[i]][_colSlices[j]] += other.matrix[i][j];
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
    public void addTo(int i, int j, NdArray other) {
        for (int _i = 0; _i < other.getShape().getRow(); _i++) {
            for (int _j = 0; _j < other.getShape().getColumn(); _j++) {
                getMatrix()[_i + i][_j + j] += other.getMatrix()[_i][_j];
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
    public NdArray clip(float min, float max) {
        float[][] _matrix = new float[shape.getRow()][shape.getColumn()];
        for (int i = 0; i < shape.getRow(); i++) {
            for (int j = 0; j < shape.getColumn(); j++) {
                if (matrix[i][j] > max) {
                    _matrix[i][j] = max;
                } else {
                    _matrix[i][j] = Math.max(matrix[i][j], min);
                }
            }
        }
        return new NdArray(_matrix);

    }

    public void fillAll(Number number) {
        for (int i = 0; i < shape.getRow(); i++) {
            for (int j = 0; j < shape.getColumn(); j++)
                matrix[i][j] = number.floatValue();
        }
    }

    /**
     * 获取第一数
     *
     * @return
     */
    public Number getNumber() {
        return matrix[0][0];
    }

    public Shape getShape() {
        return shape;
    }

    public void setShape(Shape shape) {
        this.shape = shape;
    }

    public float[][] getMatrix() {
        return matrix;
    }

    public void setMatrix(float[][] matrix) {
        this.matrix = matrix;
    }

    /**
     * 表示NdArray的形状
     */
    @Override
    public String toString() {
        return "NdArray{" + "shape=" + shape + ", matrix=" + Arrays.deepToString(matrix) + '}';
    }

}
