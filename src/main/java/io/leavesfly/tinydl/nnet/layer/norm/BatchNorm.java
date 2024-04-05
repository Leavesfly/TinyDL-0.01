package io.leavesfly.tinydl.nnet.layer.norm;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Layer;
import io.leavesfly.tinydl.nnet.Parameter;
import io.leavesfly.tinydl.utils.Config;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * 批量归一化
 */
public class BatchNorm extends Layer {

    private Parameter gammaParam;
    private Parameter betaParam;

    private NdArray runningMean;
    private NdArray runningVar;

    // backward时使用的中间数据
    private int batch_size;
    private NdArray xc;
    private NdArray xn;
    private NdArray std;

    private float momentum = 0.9f;

    public BatchNorm(String _name, Shape _xInputShape, Shape _yOutputShape) {
        super(_name, _xInputShape, _yOutputShape);
    }

    public BatchNorm(String _name, int xInputRow) {
        super(_name, new Shape(-1, xInputRow), new Shape(-1, xInputRow));

        NdArray gammaNdArray = NdArray.ones(new Shape(1, xInputRow));
        gammaParam = new Parameter(gammaNdArray);
        gammaParam.setName("gamma");
        addParam(gammaParam.getName(), gammaParam);

        NdArray betaNdArray = NdArray.zeros(new Shape(1, xInputRow));
        betaParam = new Parameter(betaNdArray);
        betaParam.setName("beta");
        addParam(betaParam.getName(), betaParam);

    }

    @Override
    public void init() {

    }

    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray input = inputs[0];
        if (Objects.isNull(runningMean)) {
            runningMean = NdArray.zeros(new Shape(1, input.getShape().getColumn()));
            runningVar = NdArray.zeros(new Shape(1, input.getShape().getColumn()));
        }

        NdArray xn = null;
        if (Config.train) {
            NdArray mu = input.mean(0);
            NdArray xc = input.sub(mu.broadcastTo(input.getShape()));
            NdArray var = xc.square().mean(0);
            NdArray std = var.add(NdArray.like(var.getShape(), 10e-7)).sqrt();
            xn = xc.div(std.broadcastTo(xc.getShape()));

            this.batch_size = input.getShape().getRow();
            this.xc = xc;
            this.xn = xn;
            this.std = std;

            this.runningMean = runningMean.mulNum(momentum).add(mu.mulNum(1f - momentum));
            this.runningVar = runningVar.mulNum(momentum).add(var.mulNum(1f - momentum));
        } else {
            NdArray xc = input.sub(runningMean.broadcastTo(input.getShape()));
            xn = xc.div(runningVar.add(NdArray.like(runningVar.getShape(), 10e-7)).sqrt().broadcastTo(xc.getShape()));
        }
        return xn.mul(gammaParam.getValue().broadcastTo(xn.getShape())).add(betaParam.getValue().broadcastTo(xn.getShape()));
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {

        NdArray dgamma = this.xn.mul(yGrad).sum(0);
        NdArray dbeta = yGrad.sum(0);

        NdArray dxn = gammaParam.getValue().broadcastTo(xn.getShape()).mul(yGrad);
        NdArray dxc = dxn.div(std.broadcastTo(dxn.getShape()));
        NdArray dstd = dxc.mul(xc).div(std.sqrt().broadcastTo(xc.getShape())).sum(0).neg();
        NdArray dvar = dstd.mulNum(0.5f).div(std.broadcastTo(dstd.getShape()));
        dxc = dxc.add(xc.mul(dvar).mulNum(2.0 / batch_size));
        NdArray dmu = dxc.sum(0);
        NdArray dx = dxc.sub(dmu.divNum(batch_size));

        return Arrays.asList(dx, dgamma, dbeta);
    }

    @Override
    public int requireInputNum() {
        return 3;
    }


    @Override
    public Variable forward(Variable... inputs) {
        return this.call(inputs[0], gammaParam, betaParam);
    }
}
