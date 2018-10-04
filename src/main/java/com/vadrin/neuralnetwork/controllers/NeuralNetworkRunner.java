package com.vadrin.neuralnetwork.controllers;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Iterator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Controller;

import com.vadrin.neuralnetwork.commons.exceptions.InvalidInputException;
import com.vadrin.neuralnetwork.mnist.MnistImageFile;
import com.vadrin.neuralnetwork.mnist.MnistLabelFile;
import com.vadrin.neuralnetwork.models.NetworkConfig;
import com.vadrin.neuralnetwork.models.TrainingExample;
import com.vadrin.neuralnetwork.models.TrainingSet;
import com.vadrin.neuralnetwork.services.NeuralNetwork;

@Controller
public class NeuralNetworkRunner implements CommandLineRunner {

	private static final Logger log = LoggerFactory.getLogger(NeuralNetworkRunner.class);
	private static final SimpleDateFormat dateFormat = new SimpleDateFormat("HH:mm:ss");

	@Override
	public void run(String... args) throws InvalidInputException {
		log.info("Starting the CommandLineRunner at {}", dateFormat.format(new Date()));
//		NetworkConfig networkConfig = new NetworkConfig((input) -> 1d / (1d + Math.exp(-input)), 5, 4, 4, 3);
//		NeuralNetwork neuralNetwork = new NeuralNetwork(networkConfig);
//
//		double[] input = { 5, 0.3, 10, 0.7, 0.1 };
//		double[] target = { 1, 0, 0 };
//		TrainingExample trainingExample = new TrainingExample(input, target);
//		double[] input2 = { 1, 3, 10, 0.7, 0.1 };
//		double[] target2 = { 0, 0, 1 };
//		TrainingExample trainingExample2 = new TrainingExample(input2, target2);
//
//		TrainingSet trainingSet = new TrainingSet();
//		trainingSet.add(trainingExample);
//		trainingSet.add(trainingExample2);
//
//		for (int i = 0; i < 10000; i++) {
//			TrainingSet randomBatch = trainingSet.getRandomBatch();
//			neuralNetwork.train(randomBatch);
//			if (i % 100 == 0) {
//				double avgrms = neuralNetwork.processAndCompareWithTrainingSetOutput(randomBatch);
//				log.info("{}% Error.", ((int)(avgrms*100)));
//			}
//		}
		
//		log.info("Result for {} is {}", Arrays.toString(input), Arrays.toString(neuralNetwork.process(input)));
//		log.info("Result for {} is {}", Arrays.toString(input2), Arrays.toString(neuralNetwork.process(input2)));
		int[] neuronsPerLayer = {784, 70, 35, 10};
		NetworkConfig config = new NetworkConfig(neuronsPerLayer, (input) -> 1d / (1d + Math.exp(-input)), -0.5d, 0.7d, -1d, 1d, 0.3d);
		NeuralNetwork neuralNetwork = new NeuralNetwork(config);
		TrainingSet trainingSet = createTrainSet(0, 4999);
		for (int i = 0; i < 50000; i++) {
			TrainingSet randomBatch = trainingSet.getRandomBatch();
			neuralNetwork.train(randomBatch);
			if (i % 100 == 0) {
				double avgrms = neuralNetwork.processAndCompareWithTrainingSetOutput(randomBatch);
				log.info("{}% Error.", ((int)(avgrms*100)));
				if(((int)(avgrms*100))<17){
					log.info("Reached desired accuracy levels. Enought training!");
					break;
				}
			}
		}
		TrainingSet testSet = createTrainSet(5000, 9999);
		testTrainSet(neuralNetwork, testSet, 10);
		log.info("Finished the CommandLineRunner at {}", dateFormat.format(new Date()));
	}
	
	
	public TrainingSet createTrainSet(int start, int end) {

		TrainingSet set = new TrainingSet();

		try {

			//String path = new File("").getAbsolutePath();

			MnistImageFile m = new MnistImageFile("C:\\Users\\pgumma\\Downloads\\res\\res\\trainImage.idx3-ubyte", "rw");
			MnistLabelFile l = new MnistLabelFile("C:\\Users\\pgumma\\Downloads\\res\\res\\trainLabel.idx1-ubyte", "rw");

			for (int i = start; i <= end; i++) {
				if (i % 100 == 0) {
					System.out.println("prepared: " + i);
				}

				double[] input = new double[28 * 28];
				double[] output = new double[10];

				output[l.readLabel()] = 1d;
				for (int j = 0; j < 28 * 28; j++) {
					input[j] = (double) m.read() / (double) 256;
				}

				set.add(input, output);
				m.next();
				l.next();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		return set;
	}
	
	public void testTrainSet(NeuralNetwork net, TrainingSet set, int printSteps) throws InvalidInputException {
		int correct = 0;

		Iterator<TrainingExample> iterator = set.iterator();
		int i = 0;
		while (iterator.hasNext()) {
			TrainingExample thisExample = iterator.next();
			double highest = indexOfHighestValue(net.process(thisExample.getInput()));
			double actualHighest = indexOfHighestValue(thisExample.getOutput());
			if (highest == actualHighest) {
				correct++;
			}
			if (i % printSteps == 0) {
				System.out.println(i + ": " + (double) correct / (double) (i + 1));
			}
			System.out.println("Testing finished, RESULT: " + correct + " / " + set.size() + "  -> "
					+ (double) correct / (double) set.size() + " %");
			i++;
		}
	}

	public int indexOfHighestValue(double[] values) {
		int index = 0;
		for (int i = 1; i < values.length; i++) {
			if (values[i] > values[index]) {
				index = i;
			}
		}
		return index;
	}

}
