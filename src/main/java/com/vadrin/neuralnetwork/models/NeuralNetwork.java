package com.vadrin.neuralnetwork.models;

import java.util.function.DoubleFunction;

import com.vadrin.neuralnetwork.commons.exceptions.InvalidInputException;

public class NeuralNetwork {

	private int[] neuronsPerLayer;
	private DoubleFunction<Double> activationFunction;
	private double[][] neuronOutputs; // Layer, NeuronPos
	private double[][] neuronBiases; // Layer, NeuronPos
	private double[][][] networkWeights; // Layer, ThisLayerNeuronPos, PrevLayerNeuronPos

	public NeuralNetwork(int[] neuronsPerLayer, DoubleFunction<Double> activationFunction) {
		super();
		this.neuronsPerLayer = neuronsPerLayer;
		this.activationFunction = activationFunction;
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
				neuronOutputs[layerIndex][thisLayerNeuronIndex] = activationFunction.apply(temp);
			}
		}

		// Return last layer output
		return neuronOutputs[neuronsPerLayer.length - 1];
	}

}
