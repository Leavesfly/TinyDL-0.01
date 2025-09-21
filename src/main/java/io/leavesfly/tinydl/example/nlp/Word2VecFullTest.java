package io.leavesfly.tinydl.example.nlp;

import io.leavesfly.tinydl.mlearning.dataset.simple.Word2VecDataSet;
import io.leavesfly.tinydl.modality.nlp.Word2Vec;
import io.leavesfly.tinydl.ndarr.NdArray;

import java.util.Arrays;
import java.util.List;

/**
 * Word2Vec完整功能测试
 * 测试数据集类和模型的集成使用
 */
public class Word2VecFullTest {
    
    public static void main(String[] args) {
        System.out.println("=== Word2Vec 完整功能测试 ===\n");
        
        // 测试1：使用Word2VecDataSet
        testWord2VecDataSet();
        
        // 测试2：中文语料测试
        testChineseCorpus();
        
        // 测试3：负采样功能测试
        testNegativeSampling();
    }
    
    /**
     * 测试Word2VecDataSet的功能
     */
    public static void testWord2VecDataSet() {
        System.out.println("=== 测试Word2VecDataSet ===");
        
        // 准备语料库
        List<String> corpus = Arrays.asList(
            "深度", "学习", "是", "机器", "学习", "的", "一个", "分支",
            "它", "使用", "神经", "网络", "来", "模拟", "人类", "大脑",
            "词", "向量", "是", "自然", "语言", "处理", "的", "基础",
            "Word2Vec", "算法", "可以", "训练", "高质量", "的", "词", "向量",
            "Skip-gram", "模型", "预测", "上下文", "词语",
            "CBOW", "模型", "预测", "中心", "词语"
        );
        
        // 创建Skip-gram数据集
        Word2VecDataSet skipGramDataSet = new Word2VecDataSet(
            corpus, 32, 2, Word2VecDataSet.TrainingMode.SKIP_GRAM, 50
        );
        skipGramDataSet.prepare();
        skipGramDataSet.printStatistics();
        
        // 创建CBOW数据集
        Word2VecDataSet cbowDataSet = new Word2VecDataSet(
            corpus, 32, 2, Word2VecDataSet.TrainingMode.CBOW, 50
        );
        cbowDataSet.prepare();
        cbowDataSet.printStatistics();
        
        System.out.println("数据集测试完成！\n");
    }
    
    /**
     * 测试中文语料库
     */
    public static void testChineseCorpus() {
        System.out.println("=== 中文语料库测试 ===");
        
        // 创建中文示例数据集
        Word2VecDataSet chineseDataSet = Word2VecDataSet.createChineseExample(16);
        chineseDataSet.prepare();
        chineseDataSet.printStatistics();
        
        // 创建Word2Vec模型测试中文
        Word2Vec chineseWord2Vec = new Word2Vec(
            "chinese_word2vec", 
            chineseDataSet.getVocabSize(), 
            8,  // 较小的维度用于测试
            Word2Vec.TrainingMode.SKIP_GRAM,
            2,
            false,
            3
        );
        
        // 构建词汇表（使用数据集的词汇）
        List<String> chineseCorpus = Arrays.asList(
            "机器", "学习", "深度", "学习", "神经", "网络", "自然", "语言", "处理",
            "词", "向量", "Word2Vec", "Skip-gram", "CBOW", "训练"
        );
        
        chineseWord2Vec.buildVocab(chineseCorpus);
        
        // 测试词向量获取
        try {
            System.out.println("中文词向量测试：");
            for (String word : Arrays.asList("机器", "学习", "神经", "网络")) {
                if (chineseWord2Vec.getWord2idx().containsKey(word)) {
                    NdArray vector = chineseWord2Vec.getWordVector(word);
                    System.out.printf("词 '%s' 的向量维度: %s\n", word, vector.getShape());
                }
            }
        } catch (Exception e) {
            System.out.println("中文词向量测试中出现错误: " + e.getMessage());
        }
        
        System.out.println("中文语料库测试完成！\n");
    }
    
    /**
     * 测试负采样功能
     */
    public static void testNegativeSampling() {
        System.out.println("=== 负采样功能测试 ===");
        
        // 创建带负采样的Word2Vec模型
        Word2Vec negativeSamplingModel = new Word2Vec(
            "negative_sampling_model",
            30,   // 词汇表大小
            6,    // 词向量维度
            Word2Vec.TrainingMode.SKIP_GRAM,
            2,    // 窗口大小
            true, // 使用负采样
            5     // 负样本数量
        );
        
        // 准备测试语料
        List<String> testCorpus = Arrays.asList(
            "人工", "智能", "是", "未来", "科技", "发展", "的", "方向",
            "机器", "学习", "让", "计算机", "具备", "学习", "能力", 
            "深度", "学习", "通过", "多层", "神经", "网络", "实现", "复杂", "模式", "识别",
            "自然", "语言", "处理", "帮助", "计算机", "理解", "人类", "语言"
        );
        
        negativeSamplingModel.buildVocab(testCorpus);
        
        // 测试负采样功能
        System.out.println("负采样测试：");
        if (!negativeSamplingModel.getWord2idx().isEmpty()) {
            // 获取几个测试词
            String[] testWords = {"人工", "智能", "机器", "学习"};
            
            for (String word : testWords) {
                if (negativeSamplingModel.getWord2idx().containsKey(word)) {
                    int wordIdx = negativeSamplingModel.getWord2idx().get(word);
                    List<Integer> negatives = negativeSamplingModel.negativeSampling(wordIdx, 3);
                    
                    System.out.printf("词 '%s' (索引: %d) 的负样本: ", word, wordIdx);
                    for (int negIdx : negatives) {
                        String negWord = negativeSamplingModel.getIdx2word().get(negIdx);
                        System.out.printf("%s(%d) ", negWord, negIdx);
                    }
                    System.out.println();
                }
            }
        }
        
        // 测试训练样本生成
        List<Word2Vec.TrainingSample> samples = negativeSamplingModel.generateTrainingSamples(testCorpus);
        System.out.println("生成的训练样本数量: " + samples.size());
        System.out.println("前5个训练样本:");
        for (int i = 0; i < Math.min(5, samples.size()); i++) {
            Word2Vec.TrainingSample sample = samples.get(i);
            String inputWord = negativeSamplingModel.getIdx2word().get(sample.input);
            String targetWord = negativeSamplingModel.getIdx2word().get(sample.target);
            System.out.printf("  样本 %d: %s(%d) -> %s(%d)\n", 
                            i+1, inputWord, sample.input, targetWord, sample.target);
        }
        
        System.out.println("负采样功能测试完成！\n");
    }
    
    /**
     * 词向量相似度测试
     */
    public static void testWordSimilarity() {
        System.out.println("=== 词向量相似度测试 ===");
        
        // 创建更大的语料库用于相似度测试
        List<String> largCorpus = Arrays.asList(
            "计算机", "科学", "是", "研究", "计算", "系统", "和", "计算", "方法", "的", "学科",
            "人工", "智能", "是", "计算机", "科学", "的", "一个", "重要", "分支",
            "机器", "学习", "是", "人工", "智能", "的", "核心", "技术", "之一",
            "深度", "学习", "是", "机器", "学习", "的", "一种", "方法",
            "神经", "网络", "是", "深度", "学习", "的", "基础", "模型",
            "自然", "语言", "处理", "让", "计算机", "理解", "人类", "语言",
            "词", "向量", "是", "自然", "语言", "处理", "的", "重要", "工具"
        );
        
        Word2Vec similarityModel = new Word2Vec(
            "similarity_model", 100, 10, Word2Vec.TrainingMode.SKIP_GRAM, 3, false, 0
        );
        
        similarityModel.buildVocab(largCorpus);
        
        // 测试相似词查找
        try {
            String[] queryWords = {"计算机", "学习", "智能"};
            for (String queryWord : queryWords) {
                if (similarityModel.getWord2idx().containsKey(queryWord)) {
                    List<String> similar = similarityModel.mostSimilar(queryWord, 3);
                    System.out.printf("与 '%s' 最相似的词: %s\n", queryWord, similar);
                }
            }
        } catch (Exception e) {
            System.out.println("相似度测试中出现错误: " + e.getMessage());
        }
        
        System.out.println("词向量相似度测试完成！");
    }
}