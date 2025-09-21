package io.leavesfly.tinydl.mlearning;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.mlearning.inference.Predictor;
import io.leavesfly.tinydl.mlearning.inference.Translator;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Block;
import io.leavesfly.tinydl.nnet.Parameter;
import io.leavesfly.tinydl.utils.Uml;

import java.io.*;
import java.util.Map;

/**
 * 机器模型的表示
 */
public class Model implements Serializable {

    private static final long serialVersionUID = 1L;

    private String name;

    private Block block;
    
    // 模型元数据信息
    private ModelInfo modelInfo;

    public transient Variable tmpPredict;

    public Model(String _name, Block _block) {
        name = _name;
        block = _block;
        modelInfo = new ModelInfo(_name);
        initializeModelInfo();
    }
    
    /**
     * 初始化模型信息
     */
    private void initializeModelInfo() {
        if (block != null) {
            modelInfo.setInputShape(block.getInputShape());
            modelInfo.setOutputShape(block.getOutputShape());
            
            // 统计参数数量
            Map<String, Parameter> params = block.getAllParams();
            long totalParams = 0;
            for (Parameter param : params.values()) {
                totalParams += param.getValue().getShape().size();
            }
            modelInfo.setTotalParameters(totalParams);
            
            // 设置架构类型（根据Block类型推断）
            String blockClassName = block.getClass().getSimpleName();
            modelInfo.setArchitectureType(blockClassName);
        }
    }

    public void plot() {
        Shape xInputShape = block.getInputShape();
        if (xInputShape != null) {
            Shape shape = block.getInputShape();
            tmpPredict = block.layerForward(new Variable(NdArray.ones(shape)));
        }
        System.out.println(Uml.getDotGraph(tmpPredict));
    }

    public void save(File modelFile) {
        try (FileOutputStream fileOut = new FileOutputStream(modelFile); ObjectOutputStream out = new ObjectOutputStream(fileOut)) {
            out.writeObject(this);
        } catch (Exception e) {
            throw new RuntimeException("Model save error!");
        }
    }
    
    /**
     * 使用ModelSerializer保存模型（推荐使用）
     * @param filePath 保存路径
     */
    public void saveModel(String filePath) {
        ModelSerializer.saveModel(this, filePath);
    }
    
    /**
     * 保存压缩模型
     * @param filePath 保存路径
     */
    public void saveModelCompressed(String filePath) {
        ModelSerializer.saveModel(this, filePath, true);
    }
    
    /**
     * 仅保存模型参数
     * @param filePath 保存路径
     */
    public void saveParameters(String filePath) {
        ModelSerializer.saveParameters(this, filePath);
    }
    
    /**
     * 保存训练检查点
     * @param filePath 保存路径
     * @param epoch 当前训练轮次
     * @param loss 当前损失
     */
    public void saveCheckpoint(String filePath, int epoch, double loss) {
        // 更新模型信息
        if (modelInfo != null) {
            modelInfo.setTrainedEpochs(epoch);
            modelInfo.setFinalLoss(loss);
            if (loss < modelInfo.getBestLoss() || modelInfo.getBestLoss() == 0) {
                modelInfo.setBestLoss(loss);
            }
        }
        ModelSerializer.saveCheckpoint(this, epoch, loss, filePath);
    }

    public static Model load(File modelFile) {
        try (FileInputStream fileIn = new FileInputStream(modelFile); ObjectInputStream in = new ObjectInputStream(fileIn)) {
            return (Model) in.readObject();
        } catch (Exception e) {
            throw new RuntimeException("model load error!");
        }
    }
    
    /**
     * 使用ModelSerializer加载模型（推荐使用）
     * @param filePath 模型文件路径
     * @return 加载的模型
     */
    public static Model loadModel(String filePath) {
        return ModelSerializer.loadModel(filePath);
    }
    
    /**
     * 加载参数到当前模型
     * @param filePath 参数文件路径
     */
    public void loadParameters(String filePath) {
        ModelSerializer.loadParameters(this, filePath);
    }
    
    /**
     * 从检查点恢复模型
     * @param filePath 检查点文件路径
     * @return 恢复的模型
     */
    public static Model resumeFromCheckpoint(String filePath) {
        return ModelSerializer.resumeFromCheckpoint(filePath);
    }

    public void resetState() {
        block.resetState();
    }

    public Variable forward(Variable... inputs) {

        return block.layerForward(inputs);
    }

    public void clearGrads() {
        block.clearGrads();
    }

    public Map<String, Parameter> getAllParams() {
        return block.getAllParams();
    }

    public <I, O> Predictor<I, O> getPredictor(Translator<I, O> translator) {
        return new Predictor<>(translator, this);
    }

    public String getName() {
        return name;
    }

    public Block getBlock() {
        return block;
    }
    
    // =========== 模型信息相关方法 ===========
    
    /**
     * 获取模型信息
     * @return 模型信息
     */
    public ModelInfo getModelInfo() {
        return modelInfo;
    }
    
    /**
     * 设置模型信息
     * @param modelInfo 模型信息
     */
    public void setModelInfo(ModelInfo modelInfo) {
        this.modelInfo = modelInfo;
    }
    
    /**
     * 更新训练信息
     * @param epochs 训练轮次
     * @param finalLoss 最终损失
     * @param optimizer 优化器名称
     * @param learningRate 学习率
     */
    public void updateTrainingInfo(int epochs, double finalLoss, String optimizer, double learningRate) {
        if (modelInfo != null) {
            modelInfo.setTrainedEpochs(epochs);
            modelInfo.setFinalLoss(finalLoss);
            modelInfo.setOptimizerType(optimizer);
            modelInfo.setLearningRate(learningRate);
            
            if (finalLoss < modelInfo.getBestLoss() || modelInfo.getBestLoss() == 0) {
                modelInfo.setBestLoss(finalLoss);
            }
        }
    }
    
    /**
     * 添加性能指标
     * @param metricName 指标名称
     * @param value 指标值
     */
    public void addMetric(String metricName, double value) {
        if (modelInfo != null) {
            modelInfo.addMetric(metricName, value);
        }
    }
    
    /**
     * 设置模型描述
     * @param description 描述
     */
    public void setDescription(String description) {
        if (modelInfo != null) {
            modelInfo.setDescription(description);
        }
    }
    
    /**
     * 获取模型简要信息
     * @return 简要信息
     */
    public String getModelSummary() {
        if (modelInfo != null) {
            return modelInfo.getSummary();
        }
        return "Model: " + name + "\nNo detailed info available.";
    }
    
    /**
     * 获取模型详细信息
     * @return 详细信息
     */
    public String getModelDetailedInfo() {
        if (modelInfo != null) {
            return modelInfo.getDetailedInfo();
        }
        return getModelSummary();
    }
    
    /**
     * 打印模型信息
     */
    public void printModelInfo() {
        System.out.println(getModelDetailedInfo());
    }
    
    /**
     * 验证模型文件
     * @param filePath 文件路径
     * @return 是否有效
     */
    public static boolean validateModel(String filePath) {
        return ModelSerializer.validateModelFile(filePath);
    }

}
