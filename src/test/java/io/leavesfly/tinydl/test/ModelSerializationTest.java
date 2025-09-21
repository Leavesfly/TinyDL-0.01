package io.leavesfly.tinydl.test;

import io.leavesfly.tinydl.mlearning.*;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.block.MlpBlock;
import io.leavesfly.tinydl.nnet.Parameter;
import io.leavesfly.tinydl.utils.Config;

import java.io.File;
import java.util.Map;

/**
 * 模型序列化功能的单元测试
 */
public class ModelSerializationTest {
    
    public static void main(String[] args) {
        System.out.println("=== 模型序列化单元测试 ===");
        
        boolean allTestsPassed = true;
        
        try {
            allTestsPassed &= testBasicSerialization();
            allTestsPassed &= testParameterSerialization();
            allTestsPassed &= testModelInfoSerialization();
            allTestsPassed &= testCheckpointSerialization();
            allTestsPassed &= testParameterComparison();
            allTestsPassed &= testJsonExport();
            
            if (allTestsPassed) {
                System.out.println("\n✅ 所有测试通过！");
            } else {
                System.out.println("\n❌ 部分测试失败！");
            }
            
        } catch (Exception e) {
            System.err.println("\n❌ 测试过程中发生异常: " + e.getMessage());
            e.printStackTrace();
            allTestsPassed = false;
        }
        
        // 清理测试文件
        cleanupTestFiles();
        
        System.exit(allTestsPassed ? 0 : 1);
    }
    
    /**
     * 测试基本序列化功能
     */
    private static boolean testBasicSerialization() {
        System.out.println("\n🧪 测试基本序列化功能...");
        
        try {
            // 创建模型
            Model originalModel = createTestModel();
            String modelPath = "test_model.model";
            
            // 保存模型
            originalModel.saveModel(modelPath);
            System.out.println("  ✓ 模型保存成功");
            
            // 加载模型
            Model loadedModel = Model.loadModel(modelPath);
            System.out.println("  ✓ 模型加载成功");
            
            // 验证模型基本信息
            assert originalModel.getName().equals(loadedModel.getName()) : "模型名称不匹配";
            System.out.println("  ✓ 模型名称匹配");
            
            // 验证参数
            boolean parametersEqual = ParameterManager.compareParameters(originalModel, loadedModel);
            assert parametersEqual : "模型参数不匹配";
            System.out.println("  ✓ 模型参数匹配");
            
            return true;
            
        } catch (Exception e) {
            System.err.println("  ❌ 基本序列化测试失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 测试参数序列化
     */
    private static boolean testParameterSerialization() {
        System.out.println("\n🧪 测试参数序列化功能...");
        
        try {
            Model model = createTestModel();
            String paramPath = "test_params.params";
            
            // 保存参数
            model.saveParameters(paramPath);
            System.out.println("  ✓ 参数保存成功");
            
            // 创建新模型并加载参数
            Model newModel = createTestModel();
            
            // 初始化不同的参数值
            initializeRandomParameters(newModel);
            
            // 验证参数不同
            boolean paramsDifferent = !ParameterManager.compareParameters(model, newModel);
            assert paramsDifferent : "初始参数应该不同";
            System.out.println("  ✓ 初始参数不同（正确）");
            
            // 加载参数
            newModel.loadParameters(paramPath);
            System.out.println("  ✓ 参数加载成功");
            
            // 验证参数相同
            boolean paramsEqual = ParameterManager.compareParameters(model, newModel);
            assert paramsEqual : "加载后参数应该相同";
            System.out.println("  ✓ 加载后参数相同");
            
            return true;
            
        } catch (Exception e) {
            System.err.println("  ❌ 参数序列化测试失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 测试模型信息序列化
     */
    private static boolean testModelInfoSerialization() {
        System.out.println("\n🧪 测试模型信息序列化...");
        
        try {
            Model model = createTestModel();
            
            // 设置模型信息
            ModelInfo info = model.getModelInfo();
            info.setDescription("测试模型");
            info.setModelVersion("1.0.0");
            model.updateTrainingInfo(50, 0.1, "Adam", 0.001);
            model.addMetric("accuracy", 0.95);
            
            String modelPath = "test_model_with_info.model";
            
            // 保存和加载
            model.saveModel(modelPath);
            Model loadedModel = Model.loadModel(modelPath);
            
            // 验证模型信息
            ModelInfo loadedInfo = loadedModel.getModelInfo();
            assert "测试模型".equals(loadedInfo.getDescription()) : "描述不匹配";
            assert "1.0.0".equals(loadedInfo.getModelVersion()) : "版本不匹配";
            assert loadedInfo.getTrainedEpochs() == 50 : "训练轮次不匹配";
            assert Math.abs(loadedInfo.getFinalLoss() - 0.1) < 1e-6 : "损失不匹配";
            assert Math.abs(loadedInfo.getMetrics().get("accuracy") - 0.95) < 1e-6 : "指标不匹配";
            
            System.out.println("  ✓ 模型信息序列化验证通过");
            
            return true;
            
        } catch (Exception e) {
            System.err.println("  ❌ 模型信息序列化测试失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 测试检查点序列化
     */
    private static boolean testCheckpointSerialization() {
        System.out.println("\n🧪 测试检查点序列化...");
        
        try {
            Model model = createTestModel();
            String checkpointPath = "test_checkpoint.ckpt";
            
            // 保存检查点
            model.saveCheckpoint(checkpointPath, 100, 0.05);
            System.out.println("  ✓ 检查点保存成功");
            
            // 从检查点恢复
            Model restoredModel = Model.resumeFromCheckpoint(checkpointPath);
            System.out.println("  ✓ 检查点恢复成功");
            
            // 验证检查点信息
            ModelInfo restoredInfo = restoredModel.getModelInfo();
            assert restoredInfo.getTrainedEpochs() == 100 : "训练轮次不匹配";
            assert Math.abs(restoredInfo.getFinalLoss() - 0.05) < 1e-6 : "损失不匹配";
            
            System.out.println("  ✓ 检查点信息验证通过");
            
            return true;
            
        } catch (Exception e) {
            System.err.println("  ❌ 检查点序列化测试失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 测试参数比较功能
     */
    private static boolean testParameterComparison() {
        System.out.println("\n🧪 测试参数比较功能...");
        
        try {
            Model model1 = createTestModel();
            Model model2 = createTestModel();
            
            // 测试相同模型的参数比较
            boolean sameParams = ParameterManager.compareParameters(model1, model1);
            assert sameParams : "同一模型参数应该相同";
            System.out.println("  ✓ 同一模型参数比较正确");
            
            // 初始化不同的参数
            initializeRandomParameters(model2);
            boolean differentParams = !ParameterManager.compareParameters(model1, model2);
            assert differentParams : "不同初始化的模型参数应该不同";
            System.out.println("  ✓ 不同模型参数比较正确");
            
            // 测试参数复制
            int copiedCount = ParameterManager.copyParameters(model1, model2);
            assert copiedCount > 0 : "应该复制了一些参数";
            System.out.println("  ✓ 参数复制成功，复制了 " + copiedCount + " 个参数");
            
            // 复制后应该相同
            boolean afterCopyEqual = ParameterManager.compareParameters(model1, model2);
            assert afterCopyEqual : "复制后参数应该相同";
            System.out.println("  ✓ 复制后参数比较正确");
            
            return true;
            
        } catch (Exception e) {
            System.err.println("  ❌ 参数比较测试失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 测试JSON导出功能
     */
    private static boolean testJsonExport() {
        System.out.println("\n🧪 测试JSON导出功能...");
        
        try {
            Model model = createTestModel();
            
            // 设置一些模型信息
            model.setDescription("JSON导出测试模型");
            model.updateTrainingInfo(25, 0.15, "SGD", 0.01);
            model.addMetric("precision", 0.88);
            
            String jsonPath = "test_model_info.json";
            String simpleJsonPath = "test_simple_report.json";
            
            // 导出JSON
            ModelInfoExporter.exportToJson(model, jsonPath);
            System.out.println("  ✓ JSON导出成功");
            
            // 导出简单报告
            ModelInfoExporter.exportSimpleReport(model, simpleJsonPath);
            System.out.println("  ✓ 简单报告导出成功");
            
            // 验证文件存在
            assert new File(jsonPath).exists() : "JSON文件不存在";
            assert new File(simpleJsonPath).exists() : "简单报告文件不存在";
            
            System.out.println("  ✓ JSON文件验证通过");
            
            return true;
            
        } catch (Exception e) {
            System.err.println("  ❌ JSON导出测试失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 创建测试模型
     */
    private static Model createTestModel() {
        MlpBlock mlpBlock = new MlpBlock("testMLP", 1, Config.ActiveFunc.ReLU, 5, 10, 3);
        mlpBlock.init();
        return new Model("TestModel", mlpBlock);
    }
    
    /**
     * 初始化随机参数
     */
    private static void initializeRandomParameters(Model model) {
        Map<String, Parameter> params = model.getAllParams();
        for (Parameter param : params.values()) {
            NdArray randomArray = NdArray.likeRandomN(param.getValue().getShape());
            param.setValue(randomArray);
        }
    }
    
    /**
     * 清理测试文件
     */
    private static void cleanupTestFiles() {
        String[] testFiles = {
            "test_model.model",
            "test_params.params", 
            "test_model_with_info.model",
            "test_checkpoint.ckpt",
            "test_model_info.json",
            "test_simple_report.json"
        };
        
        for (String filename : testFiles) {
            File file = new File(filename);
            if (file.exists()) {
                file.delete();
            }
        }
    }
}