package com.vadrin.neuralnetwork.services;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.core.JsonGenerationException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.vadrin.neuralnetwork.commons.exceptions.InvalidInputException;
import com.vadrin.neuralnetwork.commons.exceptions.NetworkNotInitializedException;
import com.vadrin.neuralnetwork.commons.utils.ArrayUtils;
import com.vadrin.neuralnetwork.models.TrainingExample;
import com.vadrin.neuralnetwork.models.DataSet;

public class NeuralNetwork {

	private int[] neuronsPerLayer;
	private double learningRate;

	private double[][] neuronBiases; // Layer, NeuronPos
	private double[][][] networkWeights; // Layer, ThisLayerNeuronPos,
											// PrevLayerNeuronPos

	private double[][] neuronOutputs; // Layer, NeuronPos
	private double[][] neuronOutputsErrorSignal; // Layer, NeuronPos

	private static final Logger log = LoggerFactory.getLogger(NeuralNetwork.class);

	public NeuralNetwork(int[] neuronsPerLayer, double learningRate, double initialRandomBiasLower,
			double initialRandomBiasUpper, double initialRandomWeightLower, double initialRandomWeightHigher) {
		super();
		this.neuronsPerLayer = neuronsPerLayer;
		this.learningRate = learningRate;
		loadRandomWeightsAndBiases(initialRandomBiasLower, initialRandomBiasUpper, initialRandomWeightLower,
				initialRandomWeightHigher);
		initializeNetwork();
	}

	public NeuralNetwork(JsonNode networkJson) {
		super();
		ObjectMapper mapper = new ObjectMapper();
		this.neuronsPerLayer = mapper.convertValue(networkJson.get("neuronsPerLayer"), int[].class);
		this.learningRate = mapper.convertValue(networkJson.get("learningRate"), double.class);
		this.neuronBiases = mapper.convertValue(networkJson.get("neuronBiases"), double[][].class);
		this.networkWeights = mapper.convertValue(networkJson.get("networkWeights"), double[][][].class);
		initializeNetwork();
	}

	private void initializeNetwork() {
		neuronOutputs = new double[neuronsPerLayer.length][];
		neuronOutputsErrorSignal = new double[neuronsPerLayer.length][];
		for (int i = 0; i < neuronsPerLayer.length; i++) {
			neuronOutputs[i] = new double[neuronsPerLayer[i]];
			neuronOutputsErrorSignal[i] = new double[neuronsPerLayer[i]];
		}
		log.info("Constructed Network with layers config as {}, learningRate as {}", Arrays.toString(neuronsPerLayer),
				learningRate);
	}

	private void loadRandomWeightsAndBiases(double initialRandomBiasLower, double initialRandomBiasUpper,
			double initialRandomWeightLower, double initialRandomWeightHigher) {
		// Initialize all Arrays
		neuronBiases = new double[neuronsPerLayer.length][];
		networkWeights = new double[neuronsPerLayer.length][][];
		for (int i = 0; i < neuronsPerLayer.length; i++) {
			// neuronBiases[i] = new double[neuronsPerLayer[i]];
			neuronBiases[i] = ArrayUtils.createRandomArray(neuronsPerLayer[i], initialRandomBiasLower,
					initialRandomBiasUpper);
			if (i > 0) {
				// networkWeights[i] = new
				// double[neuronsPerLayer[i]][neuronsPerLayer[i - 1]];
				networkWeights[i] = ArrayUtils.createRandomArray(neuronsPerLayer[i], neuronsPerLayer[i - 1],
						initialRandomWeightLower, initialRandomWeightHigher);
			}
		}
		log.debug("Loaded Network with random weights as {}, random biases as {}", Arrays.deepToString(networkWeights),
				Arrays.deepToString(neuronBiases));
	}

	// feed forward
	public double[] process(double... networkInput) throws InvalidInputException, NetworkNotInitializedException {
		if (networkInput.length != neuronsPerLayer[0])
			throw new InvalidInputException();
		if (neuronBiases == null || networkWeights == null || neuronOutputs == null || neuronOutputsErrorSignal == null)
			throw new NetworkNotInitializedException();
		// Setting outputs for input layer
		// Although this will work - this.neuronOutputs[0] = networkInput;
		// We are doing an activate to given input so that its between 0 & 1
		// And any input can be give like 5, 10, 1000 for that matter
		for (int i = 0; i < neuronsPerLayer[0]; i++) {
			this.neuronOutputs[0][i] = applySigmiodActivationFunction(networkInput[i]);
		}

		// Setting outputs for all other layers.
		// Sigmoid(Sum(Weight * outputofPRev) + Bias)
		for (int layerIndex = 1; layerIndex < neuronsPerLayer.length; layerIndex++) {
			for (int thisLayerNeuronIndex = 0; thisLayerNeuronIndex < neuronsPerLayer[layerIndex]; thisLayerNeuronIndex++) {
				double temp = 0;
				for (int prevLayerNeuronIndex = 0; prevLayerNeuronIndex < neuronsPerLayer[layerIndex
						- 1]; prevLayerNeuronIndex++) {
					temp += networkWeights[layerIndex][thisLayerNeuronIndex][prevLayerNeuronIndex]
							* neuronOutputs[layerIndex - 1][prevLayerNeuronIndex];
				}
				temp += neuronBiases[layerIndex][thisLayerNeuronIndex];
				neuronOutputs[layerIndex][thisLayerNeuronIndex] = applySigmiodActivationFunction(temp);
			}
		}
		log.debug("Network Output for input {} is {}", Arrays.toString(networkInput),
				Arrays.toString(neuronOutputs[neuronsPerLayer.length - 1]));
		return neuronOutputs[neuronsPerLayer.length - 1];
	}

	private double applySigmiodActivationFunction(double input) {
		return 1d / (1d + Math.exp(-input));
	}

	public void train(DataSet trainingSet) throws InvalidInputException, NetworkNotInitializedException {
		Iterator<TrainingExample> iterator = trainingSet.iterator();
		while (iterator.hasNext()) {
			TrainingExample thisExample = iterator.next();
			train(thisExample.getInput(), thisExample.getOutput());
		}
		log.debug("Completed Training on given training set.");
	}

	public double processAndCompareWithTrainingSetOutput(DataSet trainingSet)
			throws InvalidInputException, NetworkNotInitializedException {
		Iterator<TrainingExample> iteratorForQualityCheck = trainingSet.iterator();
		double avgrms = 0;
		while (iteratorForQualityCheck.hasNext()) {
			TrainingExample thisExample = iteratorForQualityCheck.next();
			double[] calculatedOutput = process(thisExample.getInput());
			double rms = 0;
			for (int i = 0; i < calculatedOutput.length; i++) {
				rms += (thisExample.getOutput()[i] - calculatedOutput[i])
						* (thisExample.getOutput()[i] - calculatedOutput[i]);
			}
			avgrms += rms / calculatedOutput.length;
		}
		avgrms = avgrms / trainingSet.size();
		return Math.sqrt(avgrms);
	}

	// Backproprage
	private void train(double[] input, double[] desiredOutput)
			throws InvalidInputException, NetworkNotInitializedException {
		if (desiredOutput.length != neuronsPerLayer[neuronsPerLayer.length - 1])
			throw new InvalidInputException();

		// First feedforward this input forward
		process(input);

		// Calculate the ErrorSignal for all Neurons
		populateSigmoidErrorSignalForAllNeurons(desiredOutput);
		log.debug("Error signal of last layer for input {} is {}", Arrays.toString(input),
				Arrays.toString(neuronOutputsErrorSignal[neuronsPerLayer.length - 1]));

		// Lets calculate the negative of gradient from ErrorSignal
		calculateSigmoidGradientAndUpdateWeightsAndBiases();
	}

	private void populateSigmoidErrorSignalForAllNeurons(double[] desiredOutput) {
		// Lets calculate the ErrorSignal for output layer
		for (int neuronIndex = 0; neuronIndex < neuronsPerLayer[neuronsPerLayer.length - 1]; neuronIndex++) {
			// OutputError = (Output - Desired)
			double neuronOutputsError = neuronOutputs[neuronsPerLayer.length - 1][neuronIndex]
					- desiredOutput[neuronIndex];

			// ErrorSignal = OutputError * (output) * (1-output)
			neuronOutputsErrorSignal[neuronsPerLayer.length - 1][neuronIndex] = neuronOutputsError
					* neuronOutputs[neuronsPerLayer.length - 1][neuronIndex]
					* (1d - neuronOutputs[neuronsPerLayer.length - 1][neuronIndex]);
		}

		// Lets calcuate the ErrorSignal for all hidden layers
		for (int layerIndex = (neuronsPerLayer.length - 2); layerIndex > 0; layerIndex--) {
			for (int neuronIndex = 0; neuronIndex < neuronsPerLayer[layerIndex]; neuronIndex++) {
				// OutputError = SumFromAllNeuronsOfNextLayer(weight *
				// ErrorSignalOfThatNextLayerNeuron)
				double temp = 0;
				for (int nextLayerNeuronIndex = 0; nextLayerNeuronIndex < neuronsPerLayer[layerIndex
						+ 1]; nextLayerNeuronIndex++) {
					temp += networkWeights[layerIndex + 1][nextLayerNeuronIndex][neuronIndex]
							* neuronOutputsErrorSignal[layerIndex + 1][nextLayerNeuronIndex];
				}
				// ErrorSignal = OutputError * (output) * (1-output)
				neuronOutputsErrorSignal[layerIndex][neuronIndex] = temp * neuronOutputs[layerIndex][neuronIndex]
						* (1d - neuronOutputs[layerIndex][neuronIndex]);
			}
		}
	}

	private void calculateSigmoidGradientAndUpdateWeightsAndBiases() {
		for (int layerIndex = (neuronsPerLayer.length - 1); layerIndex > 0; layerIndex--) {
			for (int neuronIndex = 0; neuronIndex < neuronsPerLayer[layerIndex]; neuronIndex++) {
				for (int prevNeuronIndex = 0; prevNeuronIndex < neuronsPerLayer[layerIndex - 1]; prevNeuronIndex++) {
					// negativeGradientForWeights = -1 * learingRate *
					// prevLayerOutput *
					// thisLayerNeuronErrorSignal
					double negativeOfWeightGradient = -learningRate * neuronOutputs[layerIndex - 1][prevNeuronIndex]
							* neuronOutputsErrorSignal[layerIndex][neuronIndex];
					// updatedWeight = oldWeight + negativeOfGradient;
					networkWeights[layerIndex][neuronIndex][prevNeuronIndex] += negativeOfWeightGradient;
				}
				// negativeGradientForBias = -1 * learningRate *
				// thisNeuronErrorSignal;
				double negativeOfBiasGradient = -learningRate * neuronOutputsErrorSignal[layerIndex][neuronIndex];
				neuronBiases[layerIndex][neuronIndex] += negativeOfBiasGradient;
			}
		}
	}

	public int[] getNeuronsPerLayer() {
		return neuronsPerLayer;
	}

	public double getLearningRate() {
		return learningRate;
	}

	public double[][] getNeuronBiases() {
		return neuronBiases;
	}

	public double[][][] getNetworkWeights() {
		return networkWeights;
	}

	public void saveNetworkToFile(File fileToSave) throws JsonGenerationException, JsonMappingException, IOException {
		ObjectMapper mapper = new ObjectMapper();
		mapper.writeValue(fileToSave, this);
		log.debug("Saved the network to file {}", fileToSave.getAbsolutePath());
	}
	
}
