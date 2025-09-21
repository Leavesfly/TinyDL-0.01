package io.leavesfly.tinydl.test;


import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.utils.Uml;

public class TestFunc {
    public static void main(String[] args) {
        test3();
    }

    public static void tanhG() {
        Variable x = new Variable(new NdArray(1.0f)).setName("x");

        Variable y = x.tanh();
        y.setName("y");

        int inters = 5;
        for (int i = 0; i < inters; i++) {
            y = y.tanh();
        }
        y.backward();
        System.out.println(Uml.getDotGraph(y));
    }

    private static void test3() {
        Variable x = new Variable(new NdArray(1.f), "x");
        Variable y = new Variable(new NdArray(1.f), "y");

        Variable z = goldStein(x, y);
        z.setName("z");
        z.backward();
        System.out.println(Uml.getDotGraph(z));
    }

    public static Variable goldStein(Variable x, Variable y) {

        return new Variable(new NdArray(1), "").add(x).add(y).add(new Variable
                (new NdArray(1), "")).squ().mul(new Variable(new NdArray(1), "").
                sub(new Variable(new NdArray(1), "").mul(x).add(new Variable(new NdArray(3), "").
                        mul(x).squ().sub(new Variable(new NdArray(1), "").mul(y)))));

    }

}
