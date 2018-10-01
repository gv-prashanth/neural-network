package com.vadrin.neuralnetwork.controllers;

import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Controller;

import com.vadrin.neuralnetwork.models.NetworkConfig;
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
		TrainingSet trainingSet = new TrainingSet(input, target);
		for (int i = 0; i < 10000; i++) {
			neuralNetwork.train(trainingSet);
			if (i % 100 == 0) {
				double[] trainingOutput = neuralNetwork.getOutput();
				log.info("Training Output is {}", Arrays.toString(trainingOutput));
			}
		}
		double[] feedForwardOutput = neuralNetwork.process(input);
		log.info("Network Output is {}", Arrays.toString(feedForwardOutput));
		log.info("Finished the CommandLineRunner at {}", dateFormat.format(new Date()));
	}

}
