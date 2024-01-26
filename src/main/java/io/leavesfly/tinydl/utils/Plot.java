package io.leavesfly.tinydl.utils;

import java.awt.*;
import java.util.HashMap;
import java.util.Map;

import javax.swing.JFrame;

import io.leavesfly.tinydl.mlearning.dataset.ArrayDataset;
import io.leavesfly.tinydl.mlearning.dataset.simple.SpiralDateSet;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;

import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

/**
 * 绘图函数
 */
public class Plot {

    private final NumberAxis xAxis = new NumberAxis("X");
    private final NumberAxis yAxis = new NumberAxis("Y");

    private XYPlot plot;

    private int index = 0;

    private Map<Integer, Color> ColorMap = new HashMap<Integer, Color>() {
        {
            put(0, Color.RED);
            put(1, Color.BLACK);
            put(2, Color.ORANGE);
            put(3, Color.PINK);
            put(4, Color.GREEN);
            put(5, Color.BLUE);
            put(6, Color.YELLOW);
        }
    };

    public void scatter(float[] x, float[] y) {

        XYDataset dataset = createDataset(x, y, "scatter");
        XYItemRenderer scatterRenderer = new XYLineAndShapeRenderer(false, true);
        scatterRenderer.setSeriesPaint(0, ColorMap.get(index));
        if (plot == null) {
            plot = new XYPlot(dataset, xAxis, yAxis, scatterRenderer);
            index++;
        } else {
            plot.setRenderer(index, scatterRenderer);
            plot.setDataset(index, dataset);
            index++;
        }
    }


    public void scatter(ArrayDataset _dataset, int[] types) {
        if (plot == null) {
            plot = new XYPlot();
            plot.setDomainAxis(xAxis);
            plot.setRangeAxis(yAxis);
        }

        for (int i : types) {
            XYDataset dataset = createDataset("scatter" + i, _dataset, i);
            XYItemRenderer scatterRenderer = new XYLineAndShapeRenderer(false, true);
            scatterRenderer.setSeriesPaint(0, ColorMap.get(index));

            plot.setRenderer(index, scatterRenderer);
            plot.setDataset(index, dataset);
            index++;
        }
    }

    public void line(float[] x, float[] y, String name) {

        XYDataset dataset = createDataset(x, y, name);
        XYLineAndShapeRenderer lineRenderer = new XYLineAndShapeRenderer(true, true);
        lineRenderer.setSeriesPaint(0, ColorMap.get(index));
        if (plot == null) {
            plot = new XYPlot(dataset, xAxis, yAxis, lineRenderer);
            index++;
        } else {
            plot.setRenderer(index, lineRenderer);
            plot.setDataset(index, dataset);
            index++;
        }
    }

    public void show() {
        plot.setOrientation(PlotOrientation.VERTICAL);
        // 创建图表对象
        JFreeChart chart = new JFreeChart(plot);
        Dimension dimension = new Dimension(800, 700);
        // 创建图表面板
        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(dimension);

        // 创建窗口并显示图表面板
        JFrame frame = new JFrame("Scatter and LineExam Chart");
        frame.setSize(dimension);
        center(frame);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().add(chartPanel);
        frame.pack();
        frame.setVisible(true);
    }

    public static void center(JFrame jFrame) {

        // 1、获取窗体的宽和高
        int widthFrame = jFrame.getWidth();
        int heightFrame = jFrame.getHeight();
        // 2、获取屏幕的宽和高
        Toolkit defaultToolkit = Toolkit.getDefaultToolkit();
        Dimension screenSize = defaultToolkit.getScreenSize();
        double widthScreen = screenSize.getWidth();
        double heightScreen = screenSize.getHeight();
        // 3、如果窗体的尺寸超过了，则直接用屏幕的尺寸
        if (widthFrame > widthScreen) {
            widthFrame = (int) widthScreen;
        }
        if (heightFrame > heightScreen) {
            heightFrame = (int) heightScreen;
        }
        // 4、设置位置
        int positionX = (int) ((widthScreen - widthFrame) / 2);
        int positionY = (int) ((heightScreen - heightFrame) / 2);

        jFrame.setSize(new Dimension(widthFrame, heightFrame));
        jFrame.setLocation(new Point(positionX, positionY));
    }


    private static XYDataset createDataset(String name, ArrayDataset dataset, int type) {
        int size = dataset.getSize();
        XYSeries scatterSeries = new XYSeries(name);
        for (int i = 0; i < size; i++) {
            if (dataset.getYs()[i].getMatrix()[0][0] == type)
                scatterSeries.add(dataset.getXs()[i].getMatrix()[0][0], dataset.getXs()[i].getMatrix()[0][1]);
        }

        XYSeriesCollection xySeriesCollection = new XYSeriesCollection();
        xySeriesCollection.addSeries(scatterSeries);
        return xySeriesCollection;
    }

    private static XYDataset createDataset(float[] x, float[] y, String name) {
        XYSeries scatterSeries = new XYSeries(name);
        for (int i = 0; i < x.length; i++) {
            scatterSeries.add(x[i], y[i]);
        }
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(scatterSeries);
        return dataset;
    }

    public static void main(String[] args) {
        test1();
    }

    public static void test1() {
        Plot plot = new Plot();
        int[] types = new int[]{0, 1, 2};

        ArrayDataset _dataset = new SpiralDateSet(100);
        _dataset.prepare();
        plot.scatter(_dataset, types);
        plot.show();
    }

}

