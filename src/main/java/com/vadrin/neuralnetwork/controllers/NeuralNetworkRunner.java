package com.vadrin.neuralnetwork.controllers;

import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Controller;

import com.vadrin.neuralnetwork.models.NeuralNetwork;

@Controller
public class NeuralNetworkRunner implements CommandLineRunner {

	private static final Logger log = LoggerFactory.getLogger(NeuralNetworkRunner.class);
	private static final SimpleDateFormat dateFormat = new SimpleDateFormat("HH:mm:ss");

	@Override
	public void run(String... args) throws Exception {
		log.info("Starting the CommandLineRunner at {}", dateFormat.format(new Date()));
		int[] neuronsPerLayer = { 5, 4, 4, 3 };
		NeuralNetwork neuralNetwork = new NeuralNetwork(neuronsPerLayer, (input) -> 1d / (1d + Math.exp(-input)), 0.3,
				0.7, -0.5, 0.7);
		neuralNetwork.feedForward(0.2, 0.3, 0.4, 0.7, 0.1);
		double[] feedForwardOutput = neuralNetwork.getOutput();
		log.info("Network Output is {}", Arrays.toString(feedForwardOutput));
		log.info("Finished the CommandLineRunner at {}", dateFormat.format(new Date()));
	}

}
