package com.cw.leibniz.network.bp;

import java.util.Arrays;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.util.TransferFunctionType;

/**
 * 类 名: BpXorTest
 * 描 述: BP网络测试————异或
 * 作 者: wicks
 * 创 建：2018年3月29日
 * 版 本：1.0
 * 
 * 历 史: 1.0 wicks 2018年3月29日 创建
 */
public class BpXorTest {
	
	public static void main(String[] args) {
		// 训练数据
        DataSet trainingSet = new DataSet(2,1);
        trainingSet.addRow(new DataSetRow(new double[]{0, 0}, new double[]{0}));
        trainingSet.addRow(new DataSetRow(new double[]{0, 1}, new double[]{1}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 0}, new double[]{1}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 1}, new double[]{0}));
        // 2个输入数据，4个神经元隐层，1个输出
        BpNetwork myPerceptron = new BpNetwork(TransferFunctionType.SIGMOID,2,4,1);
        LearningRule learningRule = myPerceptron.getLearningRule();
        BpListener listener = new BpListener();
        learningRule.addListener(listener);
        
        // 训练神经网络
        System.out.println("Training neural network...");
        myPerceptron.learn(trainingSet);
        // 测试神经网络
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
            System.out.println(" Output: " + Arrays.toString( networkOutput) );
        }
    }
	
}
