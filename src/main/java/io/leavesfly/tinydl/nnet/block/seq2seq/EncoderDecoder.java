package io.leavesfly.tinydl.nnet.block.seq2seq;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.nnet.Block;

/**
 * 编码解码器结构
 */
public class EncoderDecoder extends Block {
    private Encoder encoder;
    private Decoder decoder;

    public EncoderDecoder(String _name, Encoder encoder, Decoder decoder) {
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
