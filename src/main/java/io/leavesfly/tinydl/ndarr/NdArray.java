package io.leavesfly.tinydl.ndarr;

import io.leavesfly.tinydl.utils.Util;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;
import java.util.function.UnaryOperator;
import java.util.function.BinaryOperator;

/**
 * 支持更高维度的数据:1，标量;2，向量;3，矩阵;等N维度
 * 经过重构优化，提供更加优雅的API和更好的性能
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
    
    // 优化的随机数生成器，避免重复创建
    private static final Random RANDOM = new Random();
    
    // 常用的数学常数
    private static final float EPSILON = 1e-7f;

    // =============================================================================
    // NdArray的创建函数 - 重构后的构造方法
    // =============================================================================
    
    /**
     * 默认构造方法
     */
    public NdArray() {
    }

    /**
     * 从标量创建
     */
    public NdArray(Number number) {
        this.shape = new Shape(1, 1);
        this.buffer = new float[1];
        this.buffer[0] = number.floatValue();
    }

    /**
     * 从数据和形状创建
     */
    public NdArray(float[] data, Shape shape) {
        validateDataShape(data.length, shape.size());
        this.shape = shape;
        this.buffer = data;
    }

    /**
     * 从一维数组创建
     */
    public NdArray(float[] data) {
        this.shape = new Shape(1, data.length);
        this.buffer = data;
    }

    /**
     * 从多维数组创建 - 使用泛型递归方法替代重复代码
     */
    public NdArray(Object data) {
        if (data instanceof float[][]) {
            initFromArray((float[][]) data);
        } else if (data instanceof float[][][]) {
            initFromArray((float[][][]) data);
        } else if (data instanceof float[][][][]) {
            initFromArray((float[][][][]) data);
        } else {
            throw new IllegalArgumentException("不支持的数组类型: " + data.getClass());
        }
    }

    /**
     * 从形状创建空数组
     */
    public NdArray(Shape shape) {
        this.shape = shape;
        this.buffer = new float[shape.size()];
    }
    
    // 优化的初始化方法
    private void initFromArray(float[][] data) {
        validateArrayDimensions(data);
        this.shape = new Shape(data.length, data[0].length);
        this.buffer = new float[shape.size()];
        flattenArray(data, this.buffer, 0);
    }
    
    private void initFromArray(float[][][] data) {
        validateArrayDimensions(data);
        this.shape = new Shape(data.length, data[0].length, data[0][0].length);
        this.buffer = new float[shape.size()];
        flattenArray(data, this.buffer, 0);
    }
    
    private void initFromArray(float[][][][] data) {
        validateArrayDimensions(data);
        this.shape = new Shape(data.length, data[0].length, data[0][0].length, data[0][0][0].length);
        this.buffer = new float[shape.size()];
        flattenArray(data, this.buffer, 0);
    }
    
    // 通用数组展平方法
    private int flattenArray(Object array, float[] buffer, int index) {
        if (array instanceof float[]) {
            float[] arr = (float[]) array;
            System.arraycopy(arr, 0, buffer, index, arr.length);
            return index + arr.length;
        } else if (array.getClass().isArray()) {
            Object[] arr = (Object[]) array;
            for (Object subArray : arr) {
                index = flattenArray(subArray, buffer, index);
            }
        }
        return index;
    }
    
    // 验证数组维度一致性
    private void validateArrayDimensions(Object array) {
        if (array == null) {
            throw new IllegalArgumentException("数组不能为null");
        }
        
        if (!array.getClass().isArray()) {
            throw new IllegalArgumentException("输入必须是数组类型");
        }
        
        // 递归验证各维度的一致性
        validateDimensionConsistency(array, 0, new java.util.ArrayList<>());
    }
    
    /**
     * 递归验证多维数组各维度大小的一致性
     * @param array 当前数组对象
     * @param depth 当前递归深度
     * @param expectedSizes 各层级的期望大小列表
     */
    private void validateDimensionConsistency(Object array, int depth, java.util.List<Integer> expectedSizes) {
        if (array == null) {
            throw new IllegalArgumentException(String.format("第%d维度包含null元素", depth));
        }
        
        if (!array.getClass().isArray()) {
            return; // 到达最底层元素
        }
        
        int length = java.lang.reflect.Array.getLength(array);
        
        // 检查当前维度的大小是否与期望一致
        if (expectedSizes.size() <= depth) {
            // 第一次遇到这个深度，记录期望大小
            expectedSizes.add(length);
        } else if (!expectedSizes.get(depth).equals(length)) {
            // 发现大小不一致
            throw new IllegalArgumentException(
                String.format("第%d维度大小不一致：期望%d，实际%d", 
                    depth, expectedSizes.get(depth), length));
        }
        
        // 检查维度不能为0
        if (length == 0) {
            throw new IllegalArgumentException(
                String.format("第%d维度大小不能为0", depth));
        }
        
        // 递归检查下一级维度
        for (int i = 0; i < length; i++) {
            Object subArray = java.lang.reflect.Array.get(array, i);
            validateDimensionConsistency(subArray, depth + 1, expectedSizes);
        }
    }
    
    // 验证数据长度与形状匹配
    private void validateDataShape(int dataLength, int shapeSize) {
        if (dataLength != shapeSize) {
            throw new IllegalArgumentException(
                String.format("数据长度 %d 与形状大小 %d 不匹配", dataLength, shapeSize));
        }
    }
    
    // 验证是否为矩阵形状
    private static void validateMatrixShape(Shape shape) {
        if (!shape.isMatrix()) {
            throw new IllegalArgumentException("操作仅适用于矩阵（二维数组）");
        }
    }

    // =============================================================================
    // 静态工厂方法 - 优化后的创建方法
    // =============================================================================
    
    /**
     * 创建全零数组
     */
    public static NdArray zeros(Shape shape) {
        return new NdArray(shape); // 默认就是全零
    }

    /**
     * 创建全一数组
     */
    public static NdArray ones(Shape shape) {
        NdArray result = new NdArray(shape);
        result.fillAll(1.0f);
        return result;
    }
    
    /**
     * 创建对角矩阵
     */
    public static NdArray eye(Shape shape) {
        validateMatrixShape(shape);
        NdArray result = new NdArray(shape);
        int minDim = Math.min(shape.getRow(), shape.getColumn());
        for (int i = 0; i < minDim; i++) {
            result.buffer[i * shape.getColumn() + i] = 1.0f;
        }
        return result;
    }

    /**
     * 创建指定值的数组
     */
    public static NdArray like(Shape shape, Number value) {
        NdArray result = new NdArray(shape);
        result.fillAll(value.floatValue());
        return result;
    }

    /**
     * 创建相同形状的数组
     */
    public NdArray like(Number value) {
        return NdArray.like(this.shape, value);
    }

    /**
     * 创建标准正态分布随机数组
     */
    public static NdArray likeRandomN(Shape shape) {
        return likeRandomN(shape, 0);
    }
    
    /**
     * 创建标准正态分布随机数组（指定种子）
     */
    public static NdArray likeRandomN(Shape shape, long seed) {
        NdArray result = new NdArray(shape);
        Random random = new Random(seed);
        for (int i = 0; i < result.buffer.length; i++) {
            result.buffer[i] = (float) random.nextGaussian();
        }
        return result;
    }

    /**
     * 创建均匀分布随机数组
     */
    public static NdArray likeRandom(float min, float max, Shape shape) {
        return likeRandom(min, max, shape, 0);
    }
    
    /**
     * 创建均匀分布随机数组（指定种子）
     */
    public static NdArray likeRandom(float min, float max, Shape shape, long seed) {
        NdArray result = new NdArray(shape);
        Random random = new Random(seed);
        for (int i = 0; i < result.buffer.length; i++) {
            result.buffer[i] = random.nextFloat() * (max - min) + min;
        }
        return result;
    }

    /**
     * 创建线性空间数组（排序后）
     */
    public static NdArray linSpace(float min, float max, int num) {
        if (num <= 0) {
            throw new IllegalArgumentException("数量必须大于0");
        }
        NdArray result = likeRandom(min, max, new Shape(1, num));
        Arrays.sort(result.buffer);
        return result;
    }

    // =============================================================================
    // 基础四则运算 - 重构后的统一模式
    // =============================================================================
    
    /**
     * 通用的二元运算方法
     */
    private NdArray binaryOperation(NdArray other, BinaryOperator<Float> operation, String operationName) {
        validateShapeCompatibility(this.shape, other.shape, operationName);
        NdArray result = new NdArray(this.shape);
        for (int i = 0; i < this.buffer.length; i++) {
            result.buffer[i] = operation.apply(this.buffer[i], other.buffer[i]);
        }
        return result;
    }
    
    /**
     * 通用的与标量运算方法
     */
    private NdArray scalarOperation(Number scalar, BinaryOperator<Float> operation) {
        NdArray result = new NdArray(this.shape);
        float scalarValue = scalar.floatValue();
        for (int i = 0; i < this.buffer.length; i++) {
            result.buffer[i] = operation.apply(this.buffer[i], scalarValue);
        }
        return result;
    }
    
    /**
     * 验证形状兼容性
     */
    private static void validateShapeCompatibility(Shape shape1, Shape shape2, String operationName) {
        if (!shape1.equals(shape2)) {
            throw new IllegalArgumentException(
                String.format("%s 操作要求形状一致：%s vs %s", operationName, shape1, shape2));
        }
    }

    /**
     * 加法，必须shape一样
     */
    public NdArray add(NdArray other) {
        return binaryOperation(other, Float::sum, "加法");
    }

    /**
     * 减法，shape必须一样
     */
    public NdArray sub(NdArray other) {
        return binaryOperation(other, (a, b) -> a - b, "减法");
    }

    /**
     * 乘法，shape必须一样
     */
    public NdArray mul(NdArray other) {
        return binaryOperation(other, (a, b) -> a * b, "乘法");
    }

    /**
     * 乘法-数字
     */
    public NdArray mulNum(Number number) {
        return scalarOperation(number, (a, b) -> a * b);
    }

    /**
     * 除法-shape必须一样
     */
    public NdArray div(NdArray other) {
        return binaryOperation(other, (a, b) -> {
            if (Math.abs(b) < EPSILON) {
                throw new ArithmeticException("除数接近0");
            }
            return a / b;
        }, "除法");
    }

    /**
     * 除法-数字
     */
    public NdArray divNum(Number number) {
        float value = number.floatValue();
        if (Math.abs(value) < EPSILON) {
            throw new ArithmeticException("除数不能为0");
        }
        return scalarOperation(number, (a, b) -> a / b);
    }

    // =============================================================================
    // 逻辑运算 - 重构后的统一模式
    // =============================================================================
    
    /**
     * 通用的一元运算方法
     */
    private NdArray unaryOperation(UnaryOperator<Float> operation) {
        NdArray result = new NdArray(this.shape);
        for (int i = 0; i < this.buffer.length; i++) {
            result.buffer[i] = operation.apply(this.buffer[i]);
        }
        return result;
    }
    
    /**
     * 通用的比较运算方法
     */
    private NdArray comparisonOperation(NdArray other, java.util.function.BiPredicate<Float, Float> comparison, String operationName) {
        validateShapeCompatibility(this.shape, other.shape, operationName);
        NdArray result = new NdArray(this.shape);
        for (int i = 0; i < this.buffer.length; i++) {
            boolean compResult = comparison.test(this.buffer[i], other.buffer[i]);
            result.buffer[i] = compResult ? 1.0f : 0.0f;
        }
        return result;
    }

    /**
     * 取反操作
     */
    public NdArray neg() {
        return unaryOperation(x -> -x);
    }

    /**
     * 绝对值
     */
    public NdArray abs() {
        return unaryOperation(Math::abs);
    }

    /**
     * 相等比较
     */
    public NdArray eq(NdArray other) {
        return comparisonOperation(other, Float::equals, "相等比较");
    }

    /**
     * 大于比较
     */
    public NdArray gt(NdArray other) {
        return comparisonOperation(other, (a, b) -> a > b, "大于比较");
    }

    /**
     * 小于比较
     */
    public NdArray lt(NdArray other) {
        return comparisonOperation(other, (a, b) -> a < b, "小于比较");
    }

    /**
     * 矩阵的比较，全元素大于才大于
     */
    public boolean isLar(NdArray other) {
        validateShapeCompatibility(this.shape, other.shape, "全元素比较");
        for (int i = 0; i < this.buffer.length; i++) {
            if (this.buffer[i] <= other.buffer[i]) {
                return false;
            }
        }
        return true;
    }

    // =============================================================================
    // 基本数学函数 - 重构后的统一模式
    // =============================================================================
    
    /**
     * 通用的数学函数运算方法
     */
    private NdArray mathOperation(java.util.function.Function<Double, Double> mathFunc) {
        NdArray result = new NdArray(this.shape);
        for (int i = 0; i < this.buffer.length; i++) {
            result.buffer[i] = mathFunc.apply((double) this.buffer[i]).floatValue();
        }
        return result;
    }

    /**
     * n次方
     */
    public NdArray pow(Number number) {
        float exponent = number.floatValue();
        return mathOperation(x -> Math.pow(x, exponent));
    }

    /**
     * 平方
     */
    public NdArray square() {
        return pow(2f);
    }

    /**
     * 平方根
     */
    public NdArray sqrt() {
        return mathOperation(Math::sqrt);
    }

    /**
     * 以e为底的指数
     */
    public NdArray exp() {
        return mathOperation(Math::exp);
    }

    /**
     * sin 函数
     */
    public NdArray sin() {
        return mathOperation(Math::sin);
    }

    /**
     * cos函数
     */
    public NdArray cos() {
        return mathOperation(Math::cos);
    }

    /**
     * tanh函数
     */
    public NdArray tanh() {
        return mathOperation(Math::tanh);
    }

    /**
     * sigmoid 函数
     */
    public NdArray sigmoid() {
        return mathOperation(x -> 1.0 / (1.0 + Math.exp(-x)));
    }

    /**
     * 以e为底的对数
     */
    public NdArray log() {
        return mathOperation(x -> {
            if (x <= 0) {
                throw new ArithmeticException("对数的输入必须大于0");
            }
            return Math.log(x);
        });
    }

    /**
     * 按行累加概率为1（优化版）
     */
    public NdArray softMax() {
        validateMatrixShape(this.shape);
        NdArray result = new NdArray(this.shape);
        
        // 使用数值稳定版本，避免指数爆炸
        for (int i = 0; i < shape.getRow(); i++) {
            // 找到该行的最大值用于数值稳定
            float maxVal = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < shape.getColumn(); j++) {
                int index = i * shape.getColumn() + j;
                maxVal = Math.max(maxVal, this.buffer[index]);
            }
            
            // 计算exp和总和
            float sum = 0f;
            for (int j = 0; j < shape.getColumn(); j++) {
                int index = i * shape.getColumn() + j;
                float expVal = (float) Math.exp(this.buffer[index] - maxVal);
                result.buffer[index] = expVal;
                sum += expVal;
            }
            
            // 归一化
            for (int j = 0; j < shape.getColumn(); j++) {
                int index = i * shape.getColumn() + j;
                result.buffer[index] /= sum;
            }
        }
        return result;
    }

    /**
     * 按元素，取最大值
     */
    public NdArray maximum(Number number) {
        float threshold = number.floatValue();
        return unaryOperation(x -> Math.max(x, threshold));
    }

    /**
     * 按元素，大于number的取1，小于取0的掩码
     */
    public NdArray mask(Number number) {
        float threshold = number.floatValue();
        return unaryOperation(x -> x > threshold ? 1.0f : 0.0f);
    }

    // =============================================================================
    // 张量的变形操作 - 重构后的优化版本
    // =============================================================================

    /**
     * 转置操作（二维矩阵）
     */
    public NdArray transpose() {
        validateMatrixShape(this.shape);
        NdArray result = new NdArray(new Shape(shape.getColumn(), shape.getRow()));
        
        for (int i = 0; i < shape.getRow(); i++) {
            for (int j = 0; j < shape.getColumn(); j++) {
                result.buffer[j * shape.getRow() + i] = this.buffer[i * shape.getColumn() + j];
            }
        }
        return result;
    }

    /**
     * 多维数组转置（指定维度顺序）
     */
    public NdArray transpose(int... order) {
        validateTransposeOrder(order);
        
        int[] newDimensions = new int[shape.dimension.length];
        for (int i = 0; i < order.length; i++) {
            newDimensions[i] = shape.dimension[order[i]];
        }
        NdArray result = new NdArray(new Shape(newDimensions));

        int[] indices = new int[shape.dimension.length];
        int totalElements = shape.size();
        
        for (int i = 0; i < totalElements; i++) {
            // 将一维索引转换为多维索引
            convertToMultiIndex(i, indices);
            
            // 计算转置后的索引
            int[] transposedIndices = new int[order.length];
            for (int j = 0; j < order.length; j++) {
                transposedIndices[j] = indices[order[j]];
            }
            
            // 复制数据
            result.set(this.buffer[i], transposedIndices);
        }
        return result;
    }
    
    /**
     * 验证转置维度顺序
     */
    private void validateTransposeOrder(int[] order) {
        if (order.length != shape.dimension.length) {
            throw new IllegalArgumentException("转置维度数量不匹配");
        }
        
        boolean[] used = new boolean[shape.dimension.length];
        for (int dim : order) {
            if (dim < 0 || dim >= shape.dimension.length || used[dim]) {
                throw new IllegalArgumentException("转置维度顺序无效");
            }
            used[dim] = true;
        }
    }
    
    /**
     * 将一维索引转换为多维索引
     */
    private void convertToMultiIndex(int linearIndex, int[] indices) {
        int remaining = linearIndex;
        for (int i = shape.dimension.length - 1; i >= 0; i--) {
            if (shape.multipliers[i] == 0) {
                indices[i] = 0;
            } else {
                indices[i] = remaining / shape.multipliers[i];
                remaining %= shape.multipliers[i];
            }
        }
    }

    /**
     * 变形操作
     */
    public NdArray reshape(Shape newShape) {
        if (this.shape.size() != newShape.size()) {
            throw new IllegalArgumentException(
                String.format("形状大小不匹配：%d vs %d", this.shape.size(), newShape.size()));
        }
        
        // 使用共享数据的视图，避免数据复制
        NdArray result = new NdArray(newShape);
        System.arraycopy(this.buffer, 0, result.buffer, 0, this.buffer.length);
        return result;
    }

    /**
     * 打平成只有一行的矩阵
     */
    public NdArray flatten() {
        return this.reshape(new Shape(1, shape.size()));
    }

    // =============================================================================
    // 统计和聚合操作 - 重构后的优化版本
    // =============================================================================
    
    /**
     * 元素累和
     */
    public NdArray sum() {
        float sum = 0f;
        for (float value : this.buffer) {
            sum += value;
        }
        return new NdArray(sum);
    }
    
    /**
     * 按轴聚合的通用方法
     */
    private NdArray axisOperation(int axis, java.util.function.Function<float[], Float> operation, String operationName) {
        validateMatrixShape(this.shape);
        validateAxis(axis);
        
        if (axis == 0) {
            // 按列操作
            NdArray result = new NdArray(new Shape(1, shape.getColumn()));
            for (int j = 0; j < shape.getColumn(); j++) {
                float[] columnData = new float[shape.getRow()];
                for (int i = 0; i < shape.getRow(); i++) {
                    columnData[i] = this.buffer[i * shape.getColumn() + j];
                }
                result.buffer[j] = operation.apply(columnData);
            }
            return result;
        } else {
            // 按行操作
            NdArray result = new NdArray(new Shape(shape.getRow(), 1));
            for (int i = 0; i < shape.getRow(); i++) {
                float[] rowData = new float[shape.getColumn()];
                for (int j = 0; j < shape.getColumn(); j++) {
                    rowData[j] = this.buffer[i * shape.getColumn() + j];
                }
                result.buffer[i] = operation.apply(rowData);
            }
            return result;
        }
    }
    
    /**
     * 验证轴参数
     */
    private void validateAxis(int axis) {
        if (axis != 0 && axis != 1) {
            throw new IllegalArgumentException("轴参数只支持 0(列) 或 1(行)");
        }
    }

    /**
     * 矩阵的均值
     * axis=0表示 按列
     * axis=1表示 按行
     */
    public NdArray mean(int axis) {
        return axisOperation(axis, values -> {
            float sum = 0f;
            for (float value : values) {
                sum += value;
            }
            return sum / values.length;
        }, "均值计算");
    }

    /**
     * 矩阵的方差
     * axis=0表示 按列
     * axis=1表示 按行
     */
    public NdArray var(int axis) {
        return axisOperation(axis, values -> {
            // 计算均值
            float mean = 0f;
            for (float value : values) {
                mean += value;
            }
            mean /= values.length;
            
            // 计算方差
            float variance = 0f;
            for (float value : values) {
                variance += (value - mean) * (value - mean);
            }
            return variance / values.length;
        }, "方差计算");
    }

    /**
     * 按轴累和
     */
    public NdArray sum(int axis) {
        return axisOperation(axis, values -> {
            float sum = 0f;
            for (float value : values) {
                sum += value;
            }
            return sum;
        }, "累和计算");
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
     * 在最小最大中的值（优化版）
     */
    public NdArray clip(float min, float max) {
        if (min > max) {
            throw new IllegalArgumentException("最小值不能大于最大值");
        }
        return unaryOperation(x -> Math.max(min, Math.min(max, x)));
    }

    /**
     * 用number的值进行填充
     */
    private void fillAll(Number number) {
        float value = number.floatValue();
        Arrays.fill(this.buffer, value);
    }

    /**
     * 获取第一个元素的值
     */
    public Number getNumber() {
        return this.buffer[0];
    }

    /**
     * 获取形状
     */
    public Shape getShape() {
        return this.shape;
    }

    /**
     * 设置形状
     */
    public void setShape(Shape shape) {
        if (shape.size() != this.shape.size()) {
            throw new IllegalArgumentException("新形状大小与当前形状不匹配");
        }
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
                        for (int l = 0; l < shape.dimension[3]; l++) {
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
     * 优化的toString方法
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("NdArray{");
        sb.append("shape=").append(shape);
        sb.append(", data=");
        
        if (shape.size() <= 10) {
            // 小数组直接显示所有元素
            toStringHelper(sb, 0, new int[shape.dimension.length]);
        } else {
            // 大数组只显示前几个元素
            sb.append("[");
            for (int i = 0; i < Math.min(5, buffer.length); i++) {
                sb.append(String.format("%.4f", buffer[i]));
                if (i < Math.min(4, buffer.length - 1)) {
                    sb.append(", ");
                }
            }
            if (buffer.length > 5) {
                sb.append(", ..., ").append(String.format("%.4f", buffer[buffer.length - 1]));
            }
            sb.append("]");
        }
        
        sb.append("}");
        return sb.toString();
    }

    /**
     * 递归构建多维数组的字符串表示
     */
    private void toStringHelper(StringBuilder sb, int dimIndex, int[] indices) {
        if (dimIndex == shape.dimension.length) {
            sb.append(String.format("%.4f", get(indices)));
            return;
        }

        sb.append("[");
        for (int i = 0; i < shape.dimension[dimIndex]; i++) {
            indices[dimIndex] = i;
            toStringHelper(sb, dimIndex + 1, indices);
            if (i < shape.dimension[dimIndex] - 1) {
                sb.append(", ");
                if (dimIndex == shape.dimension.length - 2) {
                    sb.append("\n ");
                }
            }
        }
        sb.append("]");
        
        if (dimIndex == 0) {
            sb.append("\n");
        }
    }

    /**
     * 优化的equals方法
     */
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        NdArray other = (NdArray) obj;
        if (!this.shape.equals(other.shape)) return false;
        
        return Arrays.equals(this.buffer, other.buffer);
    }
    
    /**
     * 优化的hashCode方法
     */
    @Override
    public int hashCode() {
        int result = shape.hashCode();
        result = 31 * result + Arrays.hashCode(buffer);
        return result;
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
