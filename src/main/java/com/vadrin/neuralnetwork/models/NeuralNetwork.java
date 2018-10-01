package com.vadrin.neuralnetwork.models;

import java.util.function.DoubleFunction;

import com.vadrin.neuralnetwork.commons.exceptions.InvalidInputException;

public class NeuralNetwork {

	private int[] neuronsPerLayer;
	private DoubleFunction<Double> activationFunction;

	private double[][] neuronOutputs; // Layer, NeuronPos
	private double[][] neuronBiases; // Layer, NeuronPos
	private double[][][] networkWeights; // Layer, ThisLayerNeuronPos, PrevLayerNeuronPos

	private double[][] neuronOutputsErrorSignal; // Layer, NeuronPos

	public NeuralNetwork(int[] neuronsPerLayer, DoubleFunction<Double> activationFunction,
			double initialRandomBiasLower, double initialRandomBiasUpper, double initialRandomWeightLower,
			double initialRandomWeightHigher) {
		super();
		this.neuronsPerLayer = neuronsPerLayer;
		this.activationFunction = activationFunction;
		neuronOutputs = new double[neuronsPerLayer.length][];
		neuronOutputsErrorSignal = new double[neuronsPerLayer.length][];
		neuronBiases = new double[neuronsPerLayer.length][];
		networkWeights = new double[neuronsPerLayer.length][][];
		for (int i = 0; i < neuronsPerLayer.length; i++) {
			neuronOutputs[i] = new double[neuronsPerLayer[i]];
			neuronOutputsErrorSignal[i] = new double[neuronsPerLayer[i]];
			// neuronBiases[i] = new double[neuronsPerLayer[i]];
			neuronBiases[i] = createRandomArray(neuronsPerLayer[i], initialRandomBiasLower, initialRandomBiasUpper);
			if (i > 0) {
				// networkWeights[i] = new double[neuronsPerLayer[i]][neuronsPerLayer[i - 1]];
				networkWeights[i] = createRandomArray(neuronsPerLayer[i], neuronsPerLayer[i - 1],
						initialRandomWeightLower, initialRandomWeightHigher);
			}
		}
	}

	public void feedForward(double... networkInput) throws InvalidInputException {
		if (networkInput.length != neuronsPerLayer[0])
			throw new InvalidInputException();
		if (!valuesBetweenZeroAndOne(networkInput))
			throw new InvalidInputException();

		// Setting outputs for input layer
		this.neuronOutputs[0] = networkInput;

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

	}

	public void backPropagate(double[] input, double[] desiredOutput) throws InvalidInputException {
		if (input.length != neuronsPerLayer[0] || desiredOutput.length != neuronsPerLayer[neuronsPerLayer.length - 1])
			throw new InvalidInputException();
		if (!valuesBetweenZeroAndOne(input))
			throw new InvalidInputException();

		// First feed this input forward
		feedForward(input);

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
	}

	private boolean valuesBetweenZeroAndOne(double[] input) {
		for (double thisNum : input) {
			if (thisNum < 0 || thisNum > 1)
				return false;
		}
		return true;
	}

	public double[] getOutput() {
		// Return last layer output
		return neuronOutputs[neuronsPerLayer.length - 1];
	}

	private double[] createRandomArray(int size, double lower_bound, double upper_bound) {
		if (size < 1) {
			return null;
		}
		double[] ar = new double[size];
		for (int i = 0; i < size; i++) {
			ar[i] = randomValue(lower_bound, upper_bound);
		}
		return ar;
	}

	private double[][] createRandomArray(int sizeX, int sizeY, double lower_bound, double upper_bound) {
		if (sizeX < 1 || sizeY < 1) {
			return null;
		}
		double[][] ar = new double[sizeX][sizeY];
		for (int i = 0; i < sizeX; i++) {
			ar[i] = createRandomArray(sizeY, lower_bound, upper_bound);
		}
		return ar;
	}

	private double randomValue(double lower_bound, double upper_bound) {
		return Math.random() * (upper_bound - lower_bound) + lower_bound;
	}

}
