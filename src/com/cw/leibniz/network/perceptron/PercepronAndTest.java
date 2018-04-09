package com.cw.leibniz.network.perceptron;

import java.util.Arrays;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;

/**
 * 类 名: PercepronAndTest
 * 描 述: 感知机测试————与
 * 作 者: wicks
 * 创 建：2018年3月29日
 * 版 本：1.0
 * 
 * 历 史: 1.0 wicks 2018年3月29日 创建
 */
public class PercepronAndTest {
	
	public static void main(String[] args) {
		// 数据集有2个输入 和一个输出
    	// 测试数据是And逻辑运行的结果
        DataSet trainingSet = new DataSet(2, 1);
        trainingSet.addRow(new DataSetRow(new double[]{0, 0}, new double[]{0}));
        trainingSet.addRow(new DataSetRow(new double[]{0, 1}, new double[]{0}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 0}, new double[]{0}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 1}, new double[]{1}));
        // 感知机有2个输入
        PerceptronNetwork myPerceptron = new PerceptronNetwork(2);

        PerceptronLearningRule learningRule =(PerceptronLearningRule) myPerceptron.getLearningRule();
        learningRule.setMaxError(0.01);
        learningRule.setMaxIterations(1000);
        PerceptronListener listener = new PerceptronListener();
        learningRule.addListener(listener);
        
        // 进行学习
        System.out.println("Training neural network...");
        myPerceptron.learn(trainingSet);

        // 测试感知机是否能给出正确输出
        System.out.println("Testing trained neural network");
        testNeuralNetwork(myPerceptron, trainingSet);
	}
	
    @SuppressWarnings("rawtypes")
	public static void testNeuralNetwork(NeuralNetwork neuralNet, DataSet testSet) {

        for(DataSetRow testSetRow : testSet.getRows()) {
            neuralNet.setInput(testSetRow.getInput());
            neuralNet.calculate();
            double[] networkOutput = neuralNet.getOutput();

            System.out.print("Input: " + Arrays.toString( testSetRow.getInput() ) );
            System.out.println("Output: " + Arrays.toString( networkOutput) );
        }
    }

}
