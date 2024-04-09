package io.leavesfly.tinydl.nnet.layer.rnn;

import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Parameter;
import io.leavesfly.tinydl.nnet.RnnLayer;
import io.leavesfly.tinydl.nnet.layer.activate.SigmoidLayer;

import java.util.List;
import java.util.Objects;

/**
 * 长短期记忆层
 */
public class LstmLayer extends RnnLayer {

    private Variable state;

    private Variable candidate;

    private int hiddeSize;

    public LstmLayer(String _name, Shape _xInputShape, Shape _yOutputShape) {

        super(_name, _xInputShape, _yOutputShape);
        hiddeSize = _yOutputShape.getColumn();

        //x2f
        NdArray initWeight = NdArray.likeRandomN(new Shape(_xInputShape.getColumn(), hiddeSize)).mulNum(Math.sqrt((double) 1 / _xInputShape.getColumn()));
        Parameter x2f = new Parameter(initWeight);
        x2f.setName("x2f");
        addParam(x2f.getName(), x2f);

        Parameter b = new Parameter(NdArray.zeros(new Shape(1, hiddeSize)));
        b.setName("x2f-b");
        addParam(b.getName(), b);

        //x2i
        initWeight = NdArray.likeRandomN(new Shape(_xInputShape.getColumn(), hiddeSize)).mulNum(Math.sqrt((double) 1 / _xInputShape.getColumn()));
        Parameter x2i = new Parameter(initWeight);
        x2i.setName("x2i");
        addParam(x2i.getName(), x2i);

        b = new Parameter(NdArray.zeros(new Shape(1, hiddeSize)));
        b.setName("x2i-b");
        addParam(b.getName(), b);

        //x2o
        initWeight = NdArray.likeRandomN(new Shape(_xInputShape.getColumn(), hiddeSize)).mulNum(Math.sqrt((double) 1 / _xInputShape.getColumn()));
        Parameter x2o = new Parameter(initWeight);
        x2o.setName("x2o");
        addParam(x2o.getName(), x2o);

        b = new Parameter(NdArray.zeros(new Shape(1, hiddeSize)));
        b.setName("x2o-b");
        addParam(b.getName(), b);


        //x2u
        initWeight = NdArray.likeRandomN(new Shape(_xInputShape.getColumn(), hiddeSize)).mulNum(Math.sqrt((double) 1 / _xInputShape.getColumn()));
        Parameter x2u = new Parameter(initWeight);
        x2u.setName("x2u");
        addParam(x2u.getName(), x2u);

        b = new Parameter(NdArray.zeros(new Shape(1, hiddeSize)));
        b.setName("x2u-b");
        addParam(b.getName(), b);


        //=======================

        //h2f
        initWeight = NdArray.likeRandomN(new Shape(hiddeSize, hiddeSize)).mulNum(Math.sqrt((double) 1 / hiddeSize));
        Parameter h2f = new Parameter(initWeight);
        h2f.setName("h2f");
        addParam(h2f.getName(), h2f);

        //h2i
        initWeight = NdArray.likeRandomN(new Shape(hiddeSize, hiddeSize)).mulNum(Math.sqrt((double) 1 / hiddeSize));
        Parameter h2i = new Parameter(initWeight);
        h2i.setName("h2i");
        addParam(h2i.getName(), h2i);

        //h2o
        initWeight = NdArray.likeRandomN(new Shape(hiddeSize, hiddeSize)).mulNum(Math.sqrt((double) 1 / hiddeSize));
        Parameter h2o = new Parameter(initWeight);
        h2o.setName("h2o");
        addParam(h2o.getName(), h2o);

        //h2u
        initWeight = NdArray.likeRandomN(new Shape(hiddeSize, hiddeSize)).mulNum(Math.sqrt((double) 1 / hiddeSize));
        Parameter h2u = new Parameter(initWeight);
        h2u.setName("h2u");
        addParam(h2u.getName(), h2u);

        resetState();
    }

    @Override
    public void resetState() {
        state = null;
        candidate = null;
    }

    @Override
    public void init() {

    }

    @Override
    public Variable layerForward(Variable... inputs) {

        Variable x = inputs[0];

        Variable fstate = null;
        Variable istate = null;
        Variable ostate = null;
        Variable ustate = null;

        if (Objects.isNull(state)) {

            fstate = x.linear(getParamBy("x2f"), getParamBy("x2f-b"));
            fstate = new SigmoidLayer("").call(fstate);
            istate = x.linear(getParamBy("x2i"), getParamBy("x2i-b"));
            istate = new SigmoidLayer("").call(istate);
            ostate = x.linear(getParamBy("x2o"), getParamBy("x2o-b"));
            ostate = new SigmoidLayer("").call(ostate);
            ustate = x.linear(getParamBy("x2u"), getParamBy("x2u-b")).tanh();

        } else {
            fstate = x.linear(getParamBy("x2f"), getParamBy("x2f-b")).add(state.linear(getParamBy("h2f"), null));
            fstate = new SigmoidLayer("").call(fstate);
            istate = x.linear(getParamBy("x2i"), getParamBy("x2i-b")).add(state.linear(getParamBy("h2i"), null));
            istate = new SigmoidLayer("").call(istate);
            ostate = x.linear(getParamBy("x2o"), getParamBy("x2o-b")).add(state.linear(getParamBy("h2o"), null));
            ostate = new SigmoidLayer("").call(ostate);
            ustate = x.linear(getParamBy("x2u"), getParamBy("x2u-b")).add(state.linear(getParamBy("h2u"), null)).tanh();
        }

        if (Objects.isNull(candidate)) {
            candidate = istate.mul(ustate);
        } else {
            candidate = fstate.mul(candidate).add(istate.mul(ustate));
        }
        state = ostate.mul(candidate.tanh());
        return state;
    }


}
