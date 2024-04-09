package io.leavesfly.tinydl.mlearning.optimize;

import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.mlearning.Model;
import io.leavesfly.tinydl.nnet.Parameter;

import java.util.HashMap;
import java.util.Map;

/**
 * Momentum 与 AdaGrad 的融合
 */
public class Adam extends Optimizer {
    private float learningRate = 0.001f;
    private float beta1 = 0.9f;
    private float beta2 = 0.999f;
    private float epsilon = 1e-8f;

    private Map<Integer, NdArray> ms;
    private Map<Integer, NdArray> vs;
    private int t = 0;

    public Adam(Model target, float _learningRate, float _beta1, float _beta2, float _epsilon) {
        super(target);
        learningRate = _learningRate;
        beta1 = _beta1;
        beta2 = _beta2;
        epsilon = _epsilon;
    }

    public Adam(Model target) {
        super(target);
        ms = new HashMap<>();
        vs = new HashMap<>();
    }

    public void update() {
        t++;
        super.update();
    }

    @Override
    public void updateOne(Parameter parameter) {

        int key = parameter.hashCode();
        if (!ms.containsKey(key)) {
            ms.put(key, NdArray.zeros(parameter.getValue().getShape()));
            vs.put(key, NdArray.zeros(parameter.getValue().getShape()));
        }
        NdArray m = ms.get(key);
        NdArray v = vs.get(key);

        NdArray grad = parameter.getGrad();

        m = m.add(grad.sub(m).mulNum(1 - beta1));
        v = v.add(grad.mul(grad).sub(v).mulNum(1 - beta2));
        ms.put(key, m);
        vs.put(key, v);

        NdArray delat = m.mulNum(lr()).div(v.pow(0.5f).add(NdArray.like(v.getShape(), epsilon)));
        parameter.setValue(parameter.getValue().sub(delat));

    }

    private float lr() {
        float fix1 = (float) (1. - Math.pow(beta1, t));
        float fix2 = (float) (1. - Math.pow(beta2, t));
        return (float) (learningRate * Math.sqrt(fix2) / fix1);
    }
}
