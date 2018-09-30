package com.vadrin.neuralnetwork.models;

import com.vadrin.neuralnetwork.commons.exceptions.InvalidInputException;

public class NeuralNetwork {

	private int[] neuronsPerLayer;
	private double[][] neuronOutputs; // Layer, NeuronPos
	private double[][] neuronBiases; // Layer, NeuronPos
	private double[][][] networkWeights; // Layer, ThisLayerNeuronPos, PrevLayerNeuronPos

	public NeuralNetwork(int... neuronsPerLayer) {
		super();
		this.neuronsPerLayer = neuronsPerLayer;
		neuronOutputs = new double[neuronsPerLayer.length][];
		neuronBiases = new double[neuronsPerLayer.length][];
		networkWeights = new double[neuronsPerLayer.length][][];
		for (int i = 0; i < neuronsPerLayer.length; i++) {
			neuronOutputs[i] = new double[neuronsPerLayer[i]];
			neuronBiases[i] = new double[neuronsPerLayer[i]];
			if (i > 0) {
				networkWeights[i] = new double[neuronsPerLayer[i]][neuronsPerLayer[i - 1]];
			}
		}
	}

	public double[] feedForward(double... networkInput) throws InvalidInputException {
		if (networkInput.length != neuronsPerLayer[0])
			throw new InvalidInputException();

		// Setting outputs for input layer
		this.neuronOutputs[0] = networkInput;

		// Setting outputs for all other layers
		for (int layerIndex = 1; layerIndex < neuronsPerLayer.length; layerIndex++) {
			for (int thisLayerNeuronIndex = 0; thisLayerNeuronIndex < neuronsPerLayer[layerIndex]; thisLayerNeuronIndex++) {
				double temp = 0;
				for (int prevLayerNeuronIndex = 0; prevLayerNeuronIndex < neuronsPerLayer[layerIndex
						- 1]; prevLayerNeuronIndex++) {
					temp += networkWeights[layerIndex][thisLayerNeuronIndex][prevLayerNeuronIndex]
							* neuronOutputs[layerIndex - 1][prevLayerNeuronIndex];
				}
				temp += neuronBiases[layerIndex][thisLayerNeuronIndex];
				neuronOutputs[layerIndex][thisLayerNeuronIndex] = sigmoid(temp);
			}
		}

		// Return last layer output
		return neuronOutputs[neuronsPerLayer.length - 1];
	}

	public double sigmoid(double input) {
		return 1d / (1d + Math.exp(-input));
	}

}
