package io.leavesfly.tinydl.modality.nlp;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.func.math.Sigmoid;
import io.leavesfly.tinydl.mlearning.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Block;
import io.leavesfly.tinydl.nnet.Parameter;
import io.leavesfly.tinydl.nnet.layer.embedd.Embedding;
import io.leavesfly.tinydl.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinydl.nnet.layer.activate.SoftMaxLayer;
import io.leavesfly.tinydl.utils.Util;

import java.util.*;

/**
 * Word2Vec词向量模型实现
 * 支持Skip-gram和CBOW两种训练模式
 * Skip-gram: 根据中心词预测上下文词
 * CBOW: 根据上下文词预测中心词
 */
public class Word2Vec extends Block {
    
    public enum TrainingMode {
        SKIP_GRAM,  // Skip-gram模式
        CBOW        // CBOW模式
    }
    
    private final int vocabSize;      // 词汇表大小
    private final int embedSize;      // 词向量维度
    private final TrainingMode mode;  // 训练模式
    private final int windowSize;     // 窗口大小
    private final boolean useNegativeSampling; // 是否使用负采样
    private final int negativeSamples; // 负样本数量
    
    // 网络组件
    private Embedding inputEmbedding;   // 输入词嵌入层
    private Embedding outputEmbedding;  // 输出词嵌入层 (用于negative sampling)
    private LinearLayer outputLayer;   // 输出线性层
    private SoftMaxLayer softmaxLayer; // Softmax层
    
    // 词汇相关
    private Map<String, Integer> word2idx;
    private Map<Integer, String> idx2word;
    private float[] wordFreq;  // 词频统计，用于负采样
    private Random random;
    
    /**
     * 构造函数
     * @param name 模型名称
     * @param vocabSize 词汇表大小
     * @param embedSize 词向量维度
     * @param mode 训练模式 (SKIP_GRAM 或 CBOW)
     * @param windowSize 上下文窗口大小
     * @param useNegativeSampling 是否使用负采样
     * @param negativeSamples 负样本数量
     */
    public Word2Vec(String name, int vocabSize, int embedSize, TrainingMode mode, 
                   int windowSize, boolean useNegativeSampling, int negativeSamples) {
        super(name, new Shape(1, 1)); // 输入形状将在运行时确定
        
        this.vocabSize = vocabSize;
        this.embedSize = embedSize;
        this.mode = mode;
        this.windowSize = windowSize;
        this.useNegativeSampling = useNegativeSampling;
        this.negativeSamples = negativeSamples;
        this.random = new Random(42);
        
        // 初始化词汇映射
        this.word2idx = new HashMap<>();
        this.idx2word = new HashMap<>();
        this.wordFreq = new float[vocabSize];
        
        initLayers();
    }
    
    /**
     * 简化构造函数，使用默认参数
     */
    public Word2Vec(String name, int vocabSize, int embedSize) {
        this(name, vocabSize, embedSize, TrainingMode.SKIP_GRAM, 2, true, 5);
    }
    
    /**
     * 初始化网络层
     */
    private void initLayers() {
        // 输入嵌入层
        inputEmbedding = new Embedding("input_embedding", vocabSize, embedSize);
        addLayer(inputEmbedding);
        
        if (useNegativeSampling) {
            // 使用负采样时，需要输出嵌入层
            outputEmbedding = new Embedding("output_embedding", vocabSize, embedSize);
            addLayer(outputEmbedding);
        } else {
            // 不使用负采样时，使用传统的线性层+softmax
            outputLayer = new LinearLayer("output_layer", embedSize, vocabSize, true);
            softmaxLayer = new SoftMaxLayer("softmax_layer");
            addLayer(outputLayer);
            addLayer(softmaxLayer);
        }
    }
    
    @Override
    public void init() {
        // 初始化所有层
        for (int i = 0; i < layers.size(); i++) {
            layers.get(i).init();
        }
    }
    
    /**
     * 构建词汇表
     * @param corpus 语料库 (词的列表)
     */
    public void buildVocab(List<String> corpus) {
        // 统计词频
        Map<String, Integer> wordCount = new HashMap<>();
        for (String word : corpus) {
            wordCount.put(word, wordCount.getOrDefault(word, 0) + 1);
        }
        
        // 按词频排序，取前vocabSize个词
        List<Map.Entry<String, Integer>> sortedWords = new ArrayList<>(wordCount.entrySet());
        sortedWords.sort((a, b) -> b.getValue().compareTo(a.getValue()));
        
        // 构建词汇映射
        word2idx.clear();
        idx2word.clear();
        int totalCount = corpus.size();
        
        for (int i = 0; i < Math.min(vocabSize, sortedWords.size()); i++) {
            String word = sortedWords.get(i).getKey();
            int count = sortedWords.get(i).getValue();
            
            word2idx.put(word, i);
            idx2word.put(i, word);
            wordFreq[i] = (float) count / totalCount;
        }
        
        System.out.println("词汇表构建完成，共 " + word2idx.size() + " 个词");
    }
    
    /**
     * 生成训练样本
     * @param corpus 语料库
     * @return 训练样本列表
     */
    public List<TrainingSample> generateTrainingSamples(List<String> corpus) {
        List<TrainingSample> samples = new ArrayList<>();
        
        // 将词转换为索引
        List<Integer> indices = new ArrayList<>();
        for (String word : corpus) {
            Integer idx = word2idx.get(word);
            if (idx != null) {
                indices.add(idx);
            }
        }
        
        // 生成训练样本
        for (int i = 0; i < indices.size(); i++) {
            for (int j = Math.max(0, i - windowSize); j <= Math.min(indices.size() - 1, i + windowSize); j++) {
                if (i != j) {
                    if (mode == TrainingMode.SKIP_GRAM) {
                        // Skip-gram: 中心词 -> 上下文词
                        samples.add(new TrainingSample(indices.get(i), indices.get(j)));
                    } else {
                        // CBOW: 上下文词 -> 中心词
                        samples.add(new TrainingSample(indices.get(j), indices.get(i)));
                    }
                }
            }
        }
        
        System.out.println("生成训练样本 " + samples.size() + " 个");
        return samples;
    }
    
    /**
     * 前向传播
     */
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];
        
        if (mode == TrainingMode.SKIP_GRAM) {
            return forwardSkipGram(input);
        } else {
            return forwardCBOW(input);
        }
    }
    
    /**
     * Skip-gram前向传播
     */
    private Variable forwardSkipGram(Variable input) {
        // 输入是中心词索引
        Variable embedded = inputEmbedding.layerForward(input);
        
        if (useNegativeSampling) {
            // 使用负采样，返回嵌入向量
            return embedded;
        } else {
            // 传统方法：嵌入 -> 线性层 -> softmax
            Variable linear = outputLayer.layerForward(embedded);
            return softmaxLayer.layerForward(linear);
        }
    }
    
    /**
     * CBOW前向传播
     */
    private Variable forwardCBOW(Variable input) {
        // 输入是上下文词索引的平均
        Variable embedded = inputEmbedding.layerForward(input);
        
        // 对于CBOW，通常需要对多个上下文词的嵌入求平均
        // 这里简化处理，假设输入已经是平均后的结果
        
        if (useNegativeSampling) {
            return embedded;
        } else {
            Variable linear = outputLayer.layerForward(embedded);
            return softmaxLayer.layerForward(linear);
        }
    }
    
    /**
     * 负采样损失计算
     */
    public Variable negativeSamplingLoss(Variable centerEmbedding, int targetWord, List<Integer> negativeWords) {
        // 正样本损失
        Variable targetEmbedding = outputEmbedding.layerForward(new Variable(new NdArray(new float[][]{{targetWord}})));
        Variable posScore = centerEmbedding.matMul(targetEmbedding.transpose());
        Variable posLoss = new Sigmoid().call(posScore).log().neg();
        
        // 负样本损失
        Variable negLoss = new Variable(new NdArray(0f));
        for (int negWord : negativeWords) {
            Variable negEmbedding = outputEmbedding.layerForward(new Variable(new NdArray(new float[][]{{negWord}})));
            Variable negScore = centerEmbedding.matMul(negEmbedding.transpose());
            Variable negSigmoid = new Sigmoid().call(negScore.neg());
            negLoss = negLoss.add(negSigmoid.log().neg());
        }
        
        return posLoss.add(negLoss);
    }
    
    /**
     * 负采样
     */
    public List<Integer> negativeSampling(int targetWord, int numSamples) {
        List<Integer> negatives = new ArrayList<>();
        
        // 构建累积概率分布（使用3/4次方平滑）
        float[] probabilities = new float[vocabSize];
        float totalProb = 0f;
        for (int i = 0; i < vocabSize; i++) {
            probabilities[i] = (float) Math.pow(wordFreq[i], 0.75);
            totalProb += probabilities[i];
        }
        
        // 归一化
        for (int i = 0; i < vocabSize; i++) {
            probabilities[i] /= totalProb;
        }
        
        // 采样
        for (int i = 0; i < numSamples; i++) {
            int negWord;
            do {
                float rand = random.nextFloat();
                float cumProb = 0f;
                negWord = 0;
                
                for (int j = 0; j < vocabSize; j++) {
                    cumProb += probabilities[j];
                    if (rand <= cumProb) {
                        negWord = j;
                        break;
                    }
                }
            } while (negWord == targetWord || negatives.contains(negWord));
            
            negatives.add(negWord);
        }
        
        return negatives;
    }
    
    /**
     * 获取词向量
     */
    public NdArray getWordVector(String word) {
        Integer idx = word2idx.get(word);
        if (idx == null) {
            throw new IllegalArgumentException("词 '" + word + "' 不在词汇表中");
        }
        
        Variable wordVar = new Variable(new NdArray(new float[][]{{idx}}));
        Variable embedding = inputEmbedding.layerForward(wordVar);
        return embedding.getValue();
    }
    
    /**
     * 获取最相似的词
     */
    public List<String> mostSimilar(String word, int topK) {
        NdArray targetVector = getWordVector(word);
        
        List<WordSimilarity> similarities = new ArrayList<>();
        
        for (Map.Entry<String, Integer> entry : word2idx.entrySet()) {
            if (!entry.getKey().equals(word)) {
                NdArray otherVector = getWordVector(entry.getKey());
                float similarity = cosineSimilarity(targetVector, otherVector);
                similarities.add(new WordSimilarity(entry.getKey(), similarity));
            }
        }
        
        similarities.sort((a, b) -> Float.compare(b.similarity, a.similarity));
        
        List<String> result = new ArrayList<>();
        for (int i = 0; i < Math.min(topK, similarities.size()); i++) {
            result.add(similarities.get(i).word);
        }
        
        return result;
    }
    
    /**
     * 计算余弦相似度
     */
    private float cosineSimilarity(NdArray vec1, NdArray vec2) {
        float dot = 0f, norm1 = 0f, norm2 = 0f;
        
        float[] arr1 = vec1.getMatrix()[0];
        float[] arr2 = vec2.getMatrix()[0];
        
        for (int i = 0; i < arr1.length; i++) {
            dot += arr1[i] * arr2[i];
            norm1 += arr1[i] * arr1[i];
            norm2 += arr2[i] * arr2[i];
        }
        
        return dot / (float) (Math.sqrt(norm1) * Math.sqrt(norm2));
    }
    
    // Getter方法
    public int getVocabSize() { return vocabSize; }
    public int getEmbedSize() { return embedSize; }
    public TrainingMode getMode() { return mode; }
    public Map<String, Integer> getWord2idx() { return word2idx; }
    public Map<Integer, String> getIdx2word() { return idx2word; }
    
    /**
     * 训练样本类
     */
    public static class TrainingSample {
        public final int input;
        public final int target;
        
        public TrainingSample(int input, int target) {
            this.input = input;
            this.target = target;
        }
    }
    
    /**
     * 词相似度类
     */
    private static class WordSimilarity {
        public final String word;
        public final float similarity;
        
        public WordSimilarity(String word, float similarity) {
            this.word = word;
            this.similarity = similarity;
        }
    }
}
