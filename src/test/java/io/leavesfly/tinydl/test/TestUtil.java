package io.leavesfly.tinydl.test;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;

public class TestUtil {
    public static void main(String[] args) {
//        test1();
//        test4();
    }

    public static void test1() {
        //        NdArray ndArray = new NdArray(new float[][]{{1, 2, 3}});
//        System.out.println(ndArray.broadcastTo(new NdArray.Shape(2,3)));

        NdArray ndArray = new NdArray(new float[][]{{1, 2, 3}, {4, 5, 6}});
//        ndArray.softMax();

//        System.out.println(ndArray.sumTo(new NdArray.Shape(1, 3)));
//        System.out.println(ndArray.softMax());
//        System.out.println(ndArray.sumTo(new NdArray.Shape(2, 1)));
        Variable x = new Variable(ndArray);
        Variable y = x.getItem(new int[]{1}, null);
        System.out.println(y.getValue());
        y.backward();
        System.out.println(x.getGrad());

    }

    public static void test2() {

        NdArray ndArray1 = new NdArray(new float[][]{{8, 8, 7}, {4, 5, 6}});
        NdArray ndArray2 = new NdArray(new float[][]{{1}, {0}});
        Variable x = new Variable(ndArray1).setName("x");
        Variable lable = new Variable(ndArray2).setName("t");


//        Variable y = x.softmaxCrossEntropy(lable);
//        y.backward();
//        System.out.println("y : " + y.getValue());
//        System.out.println("xgetGrad : " + x.getGrad());
//        x.clearGrad();
//        List<NdArray> ndArrays = Util.limitGrad(y.getCreator(), 0);
//        System.out.println("xgetGrad limit: " + ndArrays.get(0));

//        Loss loss = new SoftmaxCE();
//        Variable loss1 = loss.loss(lable, x);
//        loss1.backward();
//        System.out.println("loss1 : " + loss1.getValue());
//        System.out.println("xgetGrad : " + x.getGrad());

//        System.out.println(getDotGraph(loss1, true));

//        y.backward();
//        System.out.println(x.getGrad());

    }


    public static void test3() {

        NdArray ndArray = new NdArray(new float[][]{{1, 2, 3}, {4, 5, 6}});

        Variable x = new Variable(ndArray);
        Variable y = x.sum();
        System.out.println(y.getValue());
        y.backward();
        System.out.println(x.getGrad());

    }

//    public static void test4() {
//
//
////        NdArray ndArray = new NdArray(new float[][]{{-1, -2, 3}, {4, 5, 6}});
//        NdArray ndArray = new NdArray(new float[][]{{-1}});
//
//        Variable x = new Variable(ndArray);
//        Variable y = x.reLU();
//        System.out.println(y.getValue());
//        y.backward();
//        System.out.println(x.getGrad());
//
//        x.clearGrad();
//        Function f = new ReLU();
//        f.call(x);
//        System.out.println("x limitGrad :" + Util.limitGrad(f, 0).get(0));
//
//
//    }

}
