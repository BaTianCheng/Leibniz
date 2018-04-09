package com.cw.leibniz.network.perceptron;

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.nnet.comp.neuron.BiasNeuron;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.util.ConnectionFactory;
import org.neuroph.util.LayerFactory;
import org.neuroph.util.NeuralNetworkFactory;
import org.neuroph.util.NeuralNetworkType;
import org.neuroph.util.NeuronProperties;
import org.neuroph.util.TransferFunctionType;

/**
 * 类 名: PerceptronNetWork
 * 描 述: 感知机网络
 * 作 者: wicks
 * 创 建：2018年3月15日
 * 版 本：1.0
 * 
 * 历 史: 1.0 wicks 2018年3月15日 创建
 */
@SuppressWarnings({ "serial", "rawtypes" })
public class PerceptronNetwork extends NeuralNetwork {
	
	public PerceptronNetwork(int inputNeuronsCount) {
		this.createNetwork(inputNeuronsCount);
	}
	
	@SuppressWarnings("unchecked")
	private void createNetwork(int inputNeuronsCount) {
		 // 设置网络类别为 感知机
        this.setNetworkType(NeuralNetworkType.PERCEPTRON);

        // 输入神经元建立 ，表示输入的刺激
        NeuronProperties inputNeuronProperties = new NeuronProperties();
        inputNeuronProperties.setProperty("neuronType", InputNeuron.class);

        // 由输入神经元构成的输入层
        Layer inputLayer = LayerFactory.createLayer(inputNeuronsCount, inputNeuronProperties);
        this.addLayer(inputLayer);
        // 在输入层增加BiasNeuron，表示神经元偏置
        inputLayer.addNeuron(new BiasNeuron());
        // 传输函数是Step
        NeuronProperties outputNeuronProperties = new NeuronProperties();
        outputNeuronProperties.setProperty("transferFunction", TransferFunctionType.STEP);

        // 输出层，也就是神经元
        Layer outputLayer = LayerFactory.createLayer(1, outputNeuronProperties);
        this.addLayer(outputLayer);

        // 将输入层和输出层进行全连接
        ConnectionFactory.fullConnect(inputLayer, outputLayer);
        NeuralNetworkFactory.setDefaultIO(this);
        // 设置感知机学习算法
        this.setLearningRule(new PerceptronLearningRule());
	}

}
