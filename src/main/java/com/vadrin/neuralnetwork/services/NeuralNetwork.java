package com.vadrin.neuralnetwork.services;

import java.util.Arrays;
import java.util.function.DoubleFunction;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.vadrin.neuralnetwork.commons.exceptions.InvalidInputException;
import com.vadrin.neuralnetwork.commons.utils.ArrayUtils;
import com.vadrin.neuralnetwork.models.NetworkConfig;
import com.vadrin.neuralnetwork.models.TrainingExample;

public class NeuralNetwork {

	private int[] neuronsPerLayer;
	private DoubleFunction<Double> activationFunction;
	private double learningRate;

	private double[][] neuronOutputs; // Layer, NeuronPos
	private double[][] neuronBiases; // Layer, NeuronPos
	private double[][][] networkWeights; // Layer, ThisLayerNeuronPos,
											// PrevLayerNeuronPos

	private double[][] neuronOutputsErrorSignal; // Layer, NeuronPos

	private static final Logger log = LoggerFactory.getLogger(NeuralNetwork.class);

	public NeuralNetwork(NetworkConfig networkConfig) {
		super();
		this.neuronsPerLayer = networkConfig.getNeuronsPerLayer();
		this.activationFunction = networkConfig.getActivationFunction();
		this.learningRate = networkConfig.getLearningRate();

		// Initialize all Arrays
		neuronOutputs = new double[neuronsPerLayer.length][];
		neuronOutputsErrorSignal = new double[neuronsPerLayer.length][];
		neuronBiases = new double[neuronsPerLayer.length][];
		networkWeights = new double[neuronsPerLayer.length][][];
		for (int i = 0; i < neuronsPerLayer.length; i++) {
			neuronOutputs[i] = new double[neuronsPerLayer[i]];
			neuronOutputsErrorSignal[i] = new double[neuronsPerLayer[i]];
			// neuronBiases[i] = new double[neuronsPerLayer[i]];
			neuronBiases[i] = ArrayUtils.createRandomArray(neuronsPerLayer[i],
					networkConfig.getInitialRandomBiasLower(), networkConfig.getInitialRandomBiasUpper());
			if (i > 0) {
				// networkWeights[i] = new
				// double[neuronsPerLayer[i]][neuronsPerLayer[i - 1]];
				networkWeights[i] = ArrayUtils.createRandomArray(neuronsPerLayer[i], neuronsPerLayer[i - 1],
						networkConfig.getInitialRandomWeightLower(), networkConfig.getInitialRandomWeightHigher());
			}
		}

		log.info(
				"Constructed Network with layers config as {}, learningRate as {}, random weights as {}, random biases as {}",
				Arrays.toString(neuronsPerLayer), learningRate, Arrays.deepToString(networkWeights),
				Arrays.deepToString(neuronBiases));
	}

	public void train(TrainingExample trainingExample) throws InvalidInputException {
		train(trainingExample.getInput(), trainingExample.getOutput());
	}

	// feed forward
	public double[] process(double... networkInput) throws InvalidInputException {
		if (networkInput.length != neuronsPerLayer[0])
			throw new InvalidInputException();

		// Setting outputs for input layer
		// Although this will work - this.neuronOutputs[0] = networkInput;
		// We are doing an activate to given input so that its between 0 & 1
		// And any input can be give like 5, 10, 1000 for that matter
		for (int i = 0; i < neuronsPerLayer[0]; i++) {
			this.neuronOutputs[0][i] = activationFunction.apply(networkInput[i]);
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
				neuronOutputs[layerIndex][thisLayerNeuronIndex] = activationFunction.apply(temp);
			}
		}
		log.debug("Network Output for input {} is {}", Arrays.toString(networkInput),
				Arrays.toString(neuronOutputs[neuronsPerLayer.length - 1]));
		return neuronOutputs[neuronsPerLayer.length - 1];
	}

	// Backproprage
	private void train(double[] input, double[] desiredOutput) throws InvalidInputException {
		if (input.length != neuronsPerLayer[0] || desiredOutput.length != neuronsPerLayer[neuronsPerLayer.length - 1])
			throw new InvalidInputException();

		// First feedforward this input forward
		process(input);

		// Calculate the ErrorSignal for all Neurons
		populateErrorSignalForAllNeurons(desiredOutput);
		log.debug("Error signal of last layer for input {} is {}", Arrays.toString(input),
				Arrays.toString(neuronOutputsErrorSignal[neuronsPerLayer.length - 1]));

		// Lets calculate the negative of gradient from ErrorSignal
		calculateGradientAndUpdateWeightsAndBiases();
	}

	private void populateErrorSignalForAllNeurons(double[] desiredOutput) {
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

	private void calculateGradientAndUpdateWeightsAndBiases() {
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

}
