package io.leavesfly.tinydl.mlearning;

import io.leavesfly.tinydl.ndarr.Shape;

import java.io.Serializable;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

/**
 * 模型元数据信息类
 * 包含模型的基本信息、架构信息、训练信息等
 */
public class ModelInfo implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    // 基本信息
    private String modelName;
    private String modelVersion;
    private String frameworkVersion;
    private Date createdTime;
    private Date lastModifiedTime;
    private String description;
    
    // 架构信息
    private String architectureType;  // 如 "MLP", "CNN", "RNN", "Transformer"
    private Shape inputShape;
    private Shape outputShape;
    private int totalLayers;
    private long totalParameters;
    private Map<String, Integer> layerCounts;  // 各类型层的数量统计
    
    // 训练信息
    private int trainedEpochs;
    private double finalLoss;
    private double bestLoss;
    private String optimizerType;
    private double learningRate;
    private int batchSize;
    private String lossFunction;
    
    // 性能信息
    private Map<String, Double> metrics;  // 如 accuracy, precision, recall等
    private long trainingTimeMs;
    private String hardwareInfo;
    
    // 自定义属性
    private Map<String, Object> customProperties;
    
    public ModelInfo() {
        this.createdTime = new Date();
        this.lastModifiedTime = new Date();
        this.frameworkVersion = "TinyDL-0.01";
        this.layerCounts = new HashMap<>();
        this.metrics = new HashMap<>();
        this.customProperties = new HashMap<>();
    }
    
    public ModelInfo(String modelName) {
        this();
        this.modelName = modelName;
    }
    
    // Getter and Setter methods
    
    public String getModelName() {
        return modelName;
    }
    
    public void setModelName(String modelName) {
        this.modelName = modelName;
        updateLastModifiedTime();
    }
    
    public String getModelVersion() {
        return modelVersion;
    }
    
    public void setModelVersion(String modelVersion) {
        this.modelVersion = modelVersion;
        updateLastModifiedTime();
    }
    
    public String getFrameworkVersion() {
        return frameworkVersion;
    }
    
    public void setFrameworkVersion(String frameworkVersion) {
        this.frameworkVersion = frameworkVersion;
        updateLastModifiedTime();
    }
    
    public Date getCreatedTime() {
        return createdTime;
    }
    
    public void setCreatedTime(Date createdTime) {
        this.createdTime = createdTime;
    }
    
    public Date getLastModifiedTime() {
        return lastModifiedTime;
    }
    
    public void setLastModifiedTime(Date lastModifiedTime) {
        this.lastModifiedTime = lastModifiedTime;
    }
    
    public String getDescription() {
        return description;
    }
    
    public void setDescription(String description) {
        this.description = description;
        updateLastModifiedTime();
    }
    
    public String getArchitectureType() {
        return architectureType;
    }
    
    public void setArchitectureType(String architectureType) {
        this.architectureType = architectureType;
        updateLastModifiedTime();
    }
    
    public Shape getInputShape() {
        return inputShape;
    }
    
    public void setInputShape(Shape inputShape) {
        this.inputShape = inputShape;
        updateLastModifiedTime();
    }
    
    public Shape getOutputShape() {
        return outputShape;
    }
    
    public void setOutputShape(Shape outputShape) {
        this.outputShape = outputShape;
        updateLastModifiedTime();
    }
    
    public int getTotalLayers() {
        return totalLayers;
    }
    
    public void setTotalLayers(int totalLayers) {
        this.totalLayers = totalLayers;
        updateLastModifiedTime();
    }
    
    public long getTotalParameters() {
        return totalParameters;
    }
    
    public void setTotalParameters(long totalParameters) {
        this.totalParameters = totalParameters;
        updateLastModifiedTime();
    }
    
    public Map<String, Integer> getLayerCounts() {
        return layerCounts;
    }
    
    public void setLayerCounts(Map<String, Integer> layerCounts) {
        this.layerCounts = layerCounts;
        updateLastModifiedTime();
    }
    
    public void addLayerCount(String layerType, int count) {
        this.layerCounts.put(layerType, count);
        updateLastModifiedTime();
    }
    
    public int getTrainedEpochs() {
        return trainedEpochs;
    }
    
    public void setTrainedEpochs(int trainedEpochs) {
        this.trainedEpochs = trainedEpochs;
        updateLastModifiedTime();
    }
    
    public double getFinalLoss() {
        return finalLoss;
    }
    
    public void setFinalLoss(double finalLoss) {
        this.finalLoss = finalLoss;
        updateLastModifiedTime();
    }
    
    public double getBestLoss() {
        return bestLoss;
    }
    
    public void setBestLoss(double bestLoss) {
        this.bestLoss = bestLoss;
        updateLastModifiedTime();
    }
    
    public String getOptimizerType() {
        return optimizerType;
    }
    
    public void setOptimizerType(String optimizerType) {
        this.optimizerType = optimizerType;
        updateLastModifiedTime();
    }
    
    public double getLearningRate() {
        return learningRate;
    }
    
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
        updateLastModifiedTime();
    }
    
    public int getBatchSize() {
        return batchSize;
    }
    
    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
        updateLastModifiedTime();
    }
    
    public String getLossFunction() {
        return lossFunction;
    }
    
    public void setLossFunction(String lossFunction) {
        this.lossFunction = lossFunction;
        updateLastModifiedTime();
    }
    
    public Map<String, Double> getMetrics() {
        return metrics;
    }
    
    public void setMetrics(Map<String, Double> metrics) {
        this.metrics = metrics;
        updateLastModifiedTime();
    }
    
    public void addMetric(String metricName, double value) {
        this.metrics.put(metricName, value);
        updateLastModifiedTime();
    }
    
    public long getTrainingTimeMs() {
        return trainingTimeMs;
    }
    
    public void setTrainingTimeMs(long trainingTimeMs) {
        this.trainingTimeMs = trainingTimeMs;
        updateLastModifiedTime();
    }
    
    public String getHardwareInfo() {
        return hardwareInfo;
    }
    
    public void setHardwareInfo(String hardwareInfo) {
        this.hardwareInfo = hardwareInfo;
        updateLastModifiedTime();
    }
    
    public Map<String, Object> getCustomProperties() {
        return customProperties;
    }
    
    public void setCustomProperties(Map<String, Object> customProperties) {
        this.customProperties = customProperties;
        updateLastModifiedTime();
    }
    
    public void addCustomProperty(String key, Object value) {
        this.customProperties.put(key, value);
        updateLastModifiedTime();
    }
    
    // 工具方法
    
    private void updateLastModifiedTime() {
        this.lastModifiedTime = new Date();
    }
    
    /**
     * 获取模型的简要信息
     */
    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("Model: ").append(modelName != null ? modelName : "Unknown").append("\n");
        sb.append("Architecture: ").append(architectureType != null ? architectureType : "Unknown").append("\n");
        sb.append("Input Shape: ").append(inputShape != null ? inputShape.toString() : "Unknown").append("\n");
        sb.append("Output Shape: ").append(outputShape != null ? outputShape.toString() : "Unknown").append("\n");
        sb.append("Total Layers: ").append(totalLayers).append("\n");
        sb.append("Total Parameters: ").append(totalParameters).append("\n");
        if (trainedEpochs > 0) {
            sb.append("Trained Epochs: ").append(trainedEpochs).append("\n");
            sb.append("Final Loss: ").append(String.format("%.6f", finalLoss)).append("\n");
        }
        return sb.toString();
    }
    
    /**
     * 获取详细信息
     */
    public String getDetailedInfo() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== 模型详细信息 ===\n");
        sb.append("名称: ").append(modelName != null ? modelName : "未知").append("\n");
        sb.append("版本: ").append(modelVersion != null ? modelVersion : "未知").append("\n");
        sb.append("框架版本: ").append(frameworkVersion).append("\n");
        sb.append("创建时间: ").append(createdTime).append("\n");
        sb.append("最后修改: ").append(lastModifiedTime).append("\n");
        if (description != null) {
            sb.append("描述: ").append(description).append("\n");
        }
        
        sb.append("\n=== 架构信息 ===\n");
        sb.append("架构类型: ").append(architectureType != null ? architectureType : "未知").append("\n");
        sb.append("输入形状: ").append(inputShape != null ? inputShape.toString() : "未知").append("\n");
        sb.append("输出形状: ").append(outputShape != null ? outputShape.toString() : "未知").append("\n");
        sb.append("总层数: ").append(totalLayers).append("\n");
        sb.append("总参数量: ").append(totalParameters).append("\n");
        
        if (!layerCounts.isEmpty()) {
            sb.append("层类型统计:\n");
            for (Map.Entry<String, Integer> entry : layerCounts.entrySet()) {
                sb.append("  ").append(entry.getKey()).append(": ").append(entry.getValue()).append("\n");
            }
        }
        
        if (trainedEpochs > 0) {
            sb.append("\n=== 训练信息 ===\n");
            sb.append("训练轮次: ").append(trainedEpochs).append("\n");
            sb.append("最终损失: ").append(String.format("%.6f", finalLoss)).append("\n");
            sb.append("最佳损失: ").append(String.format("%.6f", bestLoss)).append("\n");
            if (optimizerType != null) {
                sb.append("优化器: ").append(optimizerType).append("\n");
            }
            if (learningRate > 0) {
                sb.append("学习率: ").append(learningRate).append("\n");
            }
            if (batchSize > 0) {
                sb.append("批次大小: ").append(batchSize).append("\n");
            }
            if (lossFunction != null) {
                sb.append("损失函数: ").append(lossFunction).append("\n");
            }
            if (trainingTimeMs > 0) {
                sb.append("训练时间: ").append(trainingTimeMs / 1000.0).append(" 秒\n");
            }
        }
        
        if (!metrics.isEmpty()) {
            sb.append("\n=== 性能指标 ===\n");
            for (Map.Entry<String, Double> entry : metrics.entrySet()) {
                sb.append(entry.getKey()).append(": ").append(String.format("%.4f", entry.getValue())).append("\n");
            }
        }
        
        if (!customProperties.isEmpty()) {
            sb.append("\n=== 自定义属性 ===\n");
            for (Map.Entry<String, Object> entry : customProperties.entrySet()) {
                sb.append(entry.getKey()).append(": ").append(entry.getValue()).append("\n");
            }
        }
        
        return sb.toString();
    }
    
    @Override
    public String toString() {
        return getSummary();
    }
}