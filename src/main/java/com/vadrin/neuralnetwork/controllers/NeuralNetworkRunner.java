package com.vadrin.neuralnetwork.controllers;

import java.text.SimpleDateFormat;
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
		NeuralNetwork neuralNetwork = new NeuralNetwork();
		log.info("Finished the CommandLineRunner at {}", dateFormat.format(new Date()));
	}

}
