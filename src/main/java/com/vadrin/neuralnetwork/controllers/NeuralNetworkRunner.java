package com.vadrin.neuralnetwork.controllers;

import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Iterator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Controller;

import com.vadrin.neuralnetwork.models.NetworkConfig;
import com.vadrin.neuralnetwork.models.TrainingExample;
import com.vadrin.neuralnetwork.models.TrainingSet;
import com.vadrin.neuralnetwork.services.NeuralNetwork;

@Controller
public class NeuralNetworkRunner implements CommandLineRunner {

	private static final Logger log = LoggerFactory.getLogger(NeuralNetworkRunner.class);
	private static final SimpleDateFormat dateFormat = new SimpleDateFormat("HH:mm:ss");

	@Override
	public void run(String... args) throws Exception {
		log.info("Starting the CommandLineRunner at {}", dateFormat.format(new Date()));
		NetworkConfig networkConfig = new NetworkConfig((input) -> 1d / (1d + Math.exp(-input)), 5, 4, 4, 3);
		NeuralNetwork neuralNetwork = new NeuralNetwork(networkConfig);

		double[] input = { 5, 0.3, 10, 0.7, 0.1 };
		double[] target = { 1, 0, 0 };
		TrainingExample trainingExample = new TrainingExample(input, target);
		double[] input2 = { 1, 3, 10, 0.7, 0.1 };
		double[] target2 = { 0, 0, 1 };
		TrainingExample trainingExample2 = new TrainingExample(input2, target2);

		TrainingSet<TrainingExample> trainingSet = new TrainingSet<TrainingExample>();
		trainingSet.add(trainingExample);
		trainingSet.add(trainingExample2);

		for (int i = 0; i < 10000; i++) {
			Iterator<TrainingExample> iterator = trainingSet.iterator();
			while (iterator.hasNext()) {
				neuralNetwork.train(iterator.next());
//				if (i % 100 == 0) {
//					double[] trainingOutput = neuralNetwork.getOutput();
//					log.info("Training Output is {}", Arrays.toString(trainingOutput));
//				}
			}
		}

		log.info("Network Output is {}", Arrays.toString(neuralNetwork.process(input)));
		log.info("Network Output is {}", Arrays.toString(neuralNetwork.process(input2)));
		log.info("Finished the CommandLineRunner at {}", dateFormat.format(new Date()));
	}

}
