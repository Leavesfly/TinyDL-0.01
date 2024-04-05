package io.leavesfly.tinydl.nnet.block.transformer;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.nnet.Block;
import io.leavesfly.tinydl.nnet.block.seq2seq.Decoder;
import io.leavesfly.tinydl.nnet.block.seq2seq.Encoder;


/**
 * https://aistudio.baidu.com/aistudio/projectdetail/3034732?channelType=0&channel=0
 * <p>
 * 1,Transformer 网络架构 包括 seq2seq的结构
 * 2，encode层包括了 attention层 残差层 正则层以及线性层
 * 3，decode层包括了 attention层 残差层 正则层以及线性层
 * 4，输入输出层需要 embedding
 * //todo
 */
public class Transformer extends Block {
    private Encoder encoder;
    private Decoder decoder;


    public Transformer(String _name, Encoder encoder, Decoder decoder) {
        super(_name, encoder.getInputShape(), decoder.getOutputShape());

        this.encoder = encoder;
        this.decoder = decoder;
        getLayers().add(encoder);
        getLayers().add(decoder);

    }

    @Override
    public void init() {

    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable encoderInput = inputs[0];
        Variable decoderInput = inputs[1];

        Variable state = encoder.forward(encoderInput);
        decoder.initState(state.getValue());
        return decoder.forward(decoderInput, state);
    }
}
