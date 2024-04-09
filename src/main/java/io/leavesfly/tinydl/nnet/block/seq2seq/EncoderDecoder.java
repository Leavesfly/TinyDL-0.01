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
        addLayer(encoder);
        addLayer(decoder);
    }

    @Override
    public void init() {

    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable encoderInput = inputs[0];
        Variable decoderInput = inputs[1];

        Variable state = encoder.layerForward(encoderInput);
        decoder.initState(state.getValue());
        return decoder.layerForward(decoderInput, state);

    }
}
