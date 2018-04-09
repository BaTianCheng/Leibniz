package com.cw.leibniz.network.perceptron;

import org.neuroph.core.Connection;
import org.neuroph.core.Neuron;
import org.neuroph.core.Weight;
import org.neuroph.core.learning.SupervisedLearning;

/**
 * 类 名: PerceptronLearningRule
 * 描 述: 感知机的学习规则
 * 作 者: wicks
 * 创 建：2018年3月29日
 * 版 本：1.0
 * 
 * 历 史: 1.0 wicks 2018年3月29日 创建
 */
public class PerceptronLearningRule extends SupervisedLearning {
	
	private static final long serialVersionUID = 5196088583056101358L;

	@Override
    protected void updateNetworkWeights(double[] outputError) {
        int i = 0;
        for (Neuron neuron : neuralNetwork.getOutputNeurons()) {
            neuron.setError(outputError[i]); 
            double neuronError = neuron.getError();
            // 根据所有的神经元输入 迭代学习
            for (Connection connection : neuron.getInputConnections()) {
                // 神经元的一个输入
                double input = connection.getInput();
                // 计算权值的变更
                double weightChange =  neuronError * input;
                // 更新权值
                Weight weight = connection.getWeight();
                weight.weightChange = weightChange;                
                weight.value += weightChange;
            }

            i++;
        }
    }
}
