package com.cw.leibniz.network.perceptron;

import java.util.Arrays;

import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.IterativeLearning;

/**
 * 类 名: PerceptronListener
 * 描 述: 感知机的监听器
 * 作 者: wicks
 * 创 建：2018年3月29日
 * 版 本：1.0
 * 
 * 历 史: 1.0 wicks 2018年3月29日 创建
 */
public class PerceptronListener implements LearningEventListener{
	
    @Override
    public void handleLearningEvent(LearningEvent event) {
        IterativeLearning bp = (IterativeLearning)event.getSource();
        System.out.println("iterate:"+bp.getCurrentIteration()); 
        System.out.println(Arrays.toString(bp.getNeuralNetwork().getWeights()));
    }   
}
