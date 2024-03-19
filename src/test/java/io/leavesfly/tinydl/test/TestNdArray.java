package io.leavesfly.tinydl.test;


import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.func.base.Add;
import io.leavesfly.tinydl.func.math.Squ;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.utils.Util;

public class TestNdArray {
    private static void test1() {
        Variable x = new Variable(new NdArray(1.5F), "x");
        Function f = new Squ();
        Variable y = f.call(x);
        y.backward();

        System.out.println("backward,getGrad:" + x.getGrad().getNumber().floatValue());
        System.out.println("numericalDiff,getGrad:" + Util.numericalDiff(f, x.getValue(), 0.00004F).getNumber().floatValue());
//        System.out.println("grad,getGrad:" + Util.limitGrad(f, 0.00004F).get(0).getNumber().floatValue());
    }

    private static void test2() {
        Variable x0 = new Variable(new NdArray(new float[][]{{1, 2}, {3, 4}}), "x0");
        Variable x1 = new Variable(new NdArray(new float[][]{{5, 6}, {7, 8}}), "x1");

        Variable y = x0.mul(x1).add(x0.mul(new Variable(2))).sub(x1.mul(new Variable(5)));
        y.backward();

        System.out.println("=======backward,getGrad========");
        System.out.println("y.getValue():" + y.getValue().getNumber().floatValue());
        System.out.println("x0.getGrad():" + x0.getGrad());
        System.out.println("x1.getGrad():" + x1.getGrad());

        Variable x01 = new Variable(new NdArray(new float[][]{{2, 3, 3}, {1, 4, 3}}), "x01");
        Variable x11 = new Variable(new NdArray(new float[][]{{1, 3, 4}, {1, 4, 6}}), "x11");
        Function f1 = new Add();
        Variable y1 = f1.call(x01, x11);
        y1.backward();
        System.out.println("=======grad,getGrad========");
        System.out.println(y1.getValue().getNumber().floatValue());
        System.out.println(x01.getGrad());
        System.out.println(x11.getGrad());
    }


    private static void test4() {

        Variable x = new Variable(new NdArray(new float[][]{{1, 2, 3}, {4, 5, 6}}), "");
//        Variable c = new Variable(new NdArray(new float[][]{{10, 20, 30}, {40, 50, 60}}), "");

//        Variable t = x.add(c);
        Variable y = x.sin();
        y.backward();

        System.out.println(x.getGrad());
//        System.out.println(x.getGrad());
//        System.out.println(x.getGrad());

    }

    private static void test5() {

        Variable x = new Variable(NdArray.ones(new Shape(2, 3)), "x");
        Variable w = new Variable(NdArray.ones(new Shape(3, 4)), "w");

        Variable y = x.mul(w);
        y.backward();

        System.out.println("x:");
        System.out.println(x.getValue());
        System.out.println(x.getGrad());

        System.out.println("w:");
        System.out.println(w.getValue());
        System.out.println(w.getGrad());

        System.out.println("y:");
        System.out.println(y.getValue());
        System.out.println(y.getGrad());

    }

    public static void main(String[] args) {
//        test2();
//        System.out.println(Arrays.deepToString(new float[][]{{1, 2, 3}, {4, 5, 6}}));

        NdArray ndArray = NdArray.like(new Shape( 4, 3, 2), 1);
        System.out.println(ndArray);
    }
}
