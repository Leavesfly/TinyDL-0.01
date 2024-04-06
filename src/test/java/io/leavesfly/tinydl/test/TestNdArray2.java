package io.leavesfly.tinydl.test;

import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.NdArrayUtil;
import io.leavesfly.tinydl.ndarr.Shape;
import org.junit.Assert;
import org.junit.Test;

public class TestNdArray2 {
    @Test
    public void test() {
        Shape shape = new Shape(2, 2);

        NdArray x0 = new NdArray(new float[][]{{1, 2}, {3, 4}});
        NdArray x1 = new NdArray(new float[]{5, 6, 7, 8}, shape);

        Assert.assertEquals(new NdArray(new float[]{19, 22, 43, 50}, new Shape(2, 2)), x0.dot(x1));

        NdArray x3 = new NdArray(new float[]{5, 6, 7, 8, 9, 10}, new Shape(2, 3));
        Assert.assertEquals(new NdArray(new float[]{21, 24, 27, 47, 54, 61}, new Shape(2, 3)), x0.dot(x3));

        Assert.assertEquals("4.0", String.valueOf(x0.max()));
        Assert.assertEquals(new NdArray(new float[]{1, 3}, new Shape(2, 1)), x0.min(1));
        Assert.assertEquals(new NdArray(new float[]{2, 4}, new Shape(2, 1)), x0.max(1));
        NdArray res = x0.mul(x1);

        NdArray expected = new NdArray(new float[]{5, 12, 21, 32}, shape);
        Assert.assertEquals(res.toString(), expected.toString());

        NdArray actual = x0.sub(NdArray.like(shape, 1));
        expected = new NdArray(new float[]{0, 1, 2, 3}, shape);
        Assert.assertEquals(actual.toString(), expected.toString());

        actual = actual.add(NdArray.ones(shape));
        expected = x0;
        Assert.assertEquals(actual.toString(), expected.toString());

        actual = actual.add(NdArray.zeros(shape));
        expected = x0;
        Assert.assertEquals(actual.toString(), expected.toString());

        actual = x1.sub(NdArray.eye(shape));
        expected = new NdArray(new float[]{4, 6, 7, 7}, shape);
        Assert.assertEquals(actual.toString(), expected.toString());

        actual = x0.argMax(0);
        expected = new NdArray(new float[]{1, 1}, new Shape(1, 2));
        Assert.assertEquals(actual.toString(), expected.toString());

        actual = x0.argMax(1);
        expected = new NdArray(new float[]{1, 1}, new Shape(2, 1));
        Assert.assertEquals(actual.toString(), expected.toString());

        actual = x0.sum(0);
        expected = new NdArray(new float[]{4, 6}, new Shape(1, 2));
        Assert.assertEquals(actual.toString(), expected.toString());

        actual = x0.sum(1);
        expected = new NdArray(new float[]{3, 7}, new Shape(2, 1));
        Assert.assertEquals(actual.toString(), expected.toString());

        x3 = new NdArray(new float[]{5, 6, 7, 8, 10, 11});
        x3.setShape(new Shape(3, 2));
        actual = x3.sum(0);
        expected = new NdArray(new float[]{22, 25}, new Shape(1, 2));
        Assert.assertEquals(actual.toString(), expected.toString());

        actual = x3.sum(1);
        expected = new NdArray(new float[]{11, 15, 21}, new Shape(3, 1));
        Assert.assertEquals(actual.toString(), expected.toString());

        actual = x1.clip(10, Integer.MAX_VALUE);
        expected = new NdArray(new float[]{10, 10, 10, 10}, new Shape(2, 2));
        Assert.assertEquals(actual.toString(), expected.toString());

        actual = x1.clip(Integer.MIN_VALUE, 5);
        expected = new NdArray(new float[]{5, 5, 5, 5}, new Shape(2, 2));
        Assert.assertEquals(actual.toString(), expected.toString());

        x3 = new NdArray(new float[]{5, 6, 7, 8, 10, 11}, new Shape(3, 2));
        x3.addTo(0, 0, x1);
        expected = new NdArray(new float[]{10, 12, 14, 16, 10, 11}, new Shape(3, 2));
        Assert.assertEquals(x3.toString(), expected.toString());

        x3 = new NdArray(new float[]{5, 6, 7, 8, 10, 11}, new Shape(3, 2));
        x3.addTo(1, 0, x1);
        expected = new NdArray(new float[]{5, 6, 12, 14, 17, 19}, new Shape(3, 2));
        Assert.assertEquals(x3.toString(), expected.toString());

        NdArray merged = NdArrayUtil.merge(0, x0, x1);
        expected = new NdArray(new float[]{1, 2, 5, 6, 3, 4, 7, 8}, new Shape(2, 4));
        Assert.assertEquals(merged.toString(), expected.toString());

        merged = NdArrayUtil.merge(1, x0, x1);
        expected = new NdArray(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, new Shape(4, 2));
        Assert.assertEquals(merged.toString(), expected.toString());
    }

    @Test
    public void test2() {
        Shape shape = new Shape(3, 2);
        NdArray x1 = new NdArray(shape);
        x1 = x1.like(10);
        NdArray expected = new NdArray(new float[]{10, 10, 10, 10, 10, 10}, shape);
        Assert.assertEquals(expected.toString(), x1.toString());

        NdArray actual = x1.lt(expected);
        expected = new NdArray(new float[]{0, 0, 0, 0, 0, 0}, shape);
        Assert.assertEquals(expected.toString(), actual.toString());

        actual = NdArray.linSpace(5, 15, 6);
        actual.setShape(shape);
        expected = new NdArray(new float[]{7.405364f, 8.090506f, 11.063452f, 11.374174f, 12.309677f, 13.31441f}, shape);
        Assert.assertEquals(expected.toString(), actual.toString());

        actual = x1.gt(expected);
        expected = new NdArray(new float[]{1, 1, 0, 0, 0, 0}, shape);
        Assert.assertEquals(expected.toString(), actual.toString());

        NdArray x2 = NdArray.likeRandomN(shape);
        expected = new NdArray(new float[]{0.8025331f, -0.90154606f, 2.0809207f, 0.76377076f, 0.98457456f, -1.6834123f}, shape);
        Assert.assertEquals(expected.toString(), x2.toString());

        x2 = x2.abs();
        expected = new NdArray(new float[]{0.8025331f, 0.90154606f, 2.0809207f, 0.76377076f, 0.98457456f, 1.6834123f}, shape);
        Assert.assertEquals(expected.toString(), x2.toString());

        x2 = x2.neg();
        expected = new NdArray(new float[]{-0.8025331f, -0.90154606f, -2.0809207f, -0.76377076f, -0.98457456f, -1.6834123f}, shape);
        Assert.assertEquals(expected.toString(), x2.toString());

        x2 = x2.pow(0);
        expected = new NdArray(new float[]{1, 1, 1, 1, 1, 1}, shape);
        Assert.assertEquals(expected.toString(), x2.toString());

        Assert.assertEquals((new NdArray(10)).div(new NdArray(10)), new NdArray(1));

        Assert.assertEquals(((new NdArray(10)).mulNum(10)).eq(new NdArray(100)), new NdArray(1));

        Assert.assertEquals(new NdArray(new float[]{0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f}, shape), x2.softMax());
        Assert.assertTrue(x2.isLar(x2.softMax()));
        Assert.assertEquals(new NdArray(new float[]{1f, 1f}), x2.mean(0));

        x2 = new NdArray(new float[]{5, 6}, new Shape(1, 2));
        Shape shape1 = new Shape(3, 2);
        Assert.assertEquals(new NdArray(new float[]{5f, 6f, 5f, 6f, 5f, 6f}, shape1), x2.broadcastTo(shape1));

        Assert.assertEquals(new NdArray(new float[]{5.5f}, new Shape(1, 1)), x2.mean(1));

        Assert.assertEquals(new NdArray(new float[]{5f, 6f}, new Shape(2, 1)), x2.reshape(new Shape(2, 1)));

        NdArray x4 = new NdArray(new float[]{1, 2, 3, 4, 5, 6}, new Shape(2, 3));
        Assert.assertEquals(new NdArray(new float[]{2, 3, 5, 6}, new Shape(2, 2)), x4.subNdArray(0, 2, 1, 3));

        Assert.assertEquals(new NdArray(new float[]{1, 4, 2, 5, 3, 6}, new Shape(3, 2)), x4.transpose());

        Assert.assertEquals(new NdArray(21), x4.sum());

        expected = x4.getItem(new int[]{0, 1}, new int[]{1, 1});
        Assert.assertEquals(new NdArray(new float[]{2f, 5f}, new Shape(1, 2)), expected);

        expected = x4.getItem(new int[]{0}, null);
        Assert.assertEquals(new NdArray(new float[]{1f, 2f, 3f}, new Shape(1, 3)), expected);

        expected = x4.getItem(null, new int[]{1}).mask(4);
        Assert.assertEquals(new NdArray(new float[]{0f, 1f}, new Shape(2, 1)), expected);

        expected = x4.getItem(null, new int[]{1}).maximum(4);
        Assert.assertEquals(new NdArray(new float[]{4f, 5f}, new Shape(2, 1)), expected);

        NdArray x5 = NdArray.zeros(new Shape(3, 3));
        NdArray x6 = NdArray.ones(new Shape(1, 3));
        Assert.assertEquals(NdArray.eye(new Shape(3, 3)), x5.addAt(new int[]{0, 1, 2}, new int[]{0, 1, 2}, x6));
        x6 = NdArray.ones(new Shape(3, 3));
        Assert.assertEquals(new NdArray(new float[]{0, 0, 0, 1, 1, 1, 0, 0, 0}, new Shape(3, 3)), x5.addAt(new int[]{1}, null, x6));

        Assert.assertEquals(new NdArray(new float[]{0, 1, 0, 0, 1, 0, 0, 1, 0}, new Shape(3, 3)), x5.addAt(null, new int[]{1}, x6));

    }
}
