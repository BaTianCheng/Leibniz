package com.cw.leibniz.network.bp;

import org.neuroph.core.Connection;
import org.neuroph.core.Neuron;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.IterativeLearning;

/**
 * 类 名: BpListener
 * 描 述: BP网络监听器
 * 作 者: wicks
 * 创 建：2018年3月29日
 * 版 本：1.0
 * 
 * 历 史: 1.0 wicks 2018年3月29日 创建
 */
public class BpListener implements LearningEventListener{
	
	@Override
    public void handleLearningEvent(LearningEvent event) {
        System.out.println("============");
        IterativeLearning bp = (IterativeLearning)event.getSource();
        System.out.println("iterate:"+bp.getCurrentIteration()); 
        Neuron neuron=(Neuron)bp.getNeuralNetwork().getOutputNeurons()[0];

        for(Connection conn:neuron.getInputConnections()){
            System.out.println(conn.getWeight().value);
        }
    }    

}
