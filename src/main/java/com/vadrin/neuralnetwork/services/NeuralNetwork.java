package com.vadrin.neuralnetwork.services;

import java.util.Arrays;
import java.util.Iterator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.vadrin.neuralnetwork.commons.exceptions.InvalidInputException;
import com.vadrin.neuralnetwork.commons.exceptions.NetworkNotInitializedException;
import com.vadrin.neuralnetwork.commons.utils.ArrayUtils;
import com.vadrin.neuralnetwork.models.DataSet;
import com.vadrin.neuralnetwork.models.TrainingExample;

/*
 * Reference video: https://www.youtube.com/playlist?list=PLgomWLYGNl1dL1Qsmgumhcg4HOcWZMd3k
 */

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
			double initialRandomBiasUpper, double initialRandomWeightLower, double initialRandomWeightHigher)
			throws InvalidInputException {
		super();
		loadNeuronsPerLayerAndLearningRate(neuronsPerLayer, learningRate);
		loadRandomWeightsAndBiases(initialRandomBiasLower, initialRandomBiasUpper, initialRandomWeightLower,
				initialRandomWeightHigher);
		loadEmptyOutputAndErrorSignal();
		log.info(
				"Constructed Network with layers neuronsPerLayer as {}, learningRate as {}, random weights as {}, random biases as {}",
				Arrays.toString(neuronsPerLayer), learningRate, Arrays.deepToString(networkWeights),
				Arrays.deepToString(neuronBiases));
	}
	
	public NeuralNetwork(int[] neuronsPerLayer, double learningRate, double[][] neuronBiases, double[][][] networkWeights) {
		super();
		this.neuronsPerLayer = neuronsPerLayer;
		this.learningRate = learningRate;
		this.neuronBiases = neuronBiases;
		this.networkWeights =networkWeights;
		loadEmptyOutputAndErrorSignal();
	}

	private void loadNeuronsPerLayerAndLearningRate(int[] neuronsPerLayer, double learningRate) {
		this.neuronsPerLayer = neuronsPerLayer;
		this.learningRate = learningRate;
	}

	private void loadRandomWeightsAndBiases(double initialRandomBiasLower, double initialRandomBiasUpper,
			double initialRandomWeightLower, double initialRandomWeightHigher) throws InvalidInputException {
		// Initialize all weights and biases Arrays
		neuronBiases = new double[neuronsPerLayer.length][];
		networkWeights = new double[neuronsPerLayer.length][][];
		for (int i = 0; i < neuronsPerLayer.length; i++) {
			neuronBiases[i] = ArrayUtils.createRandomArray(neuronsPerLayer[i], initialRandomBiasLower,
					initialRandomBiasUpper);
			if (i > 0) {
				networkWeights[i] = ArrayUtils.createRandomArray(neuronsPerLayer[i], neuronsPerLayer[i - 1],
						initialRandomWeightLower, initialRandomWeightHigher);
			}
		}
	}

	private void loadEmptyOutputAndErrorSignal() {
		// Initialize output and errorsignal arrays
		neuronOutputs = new double[neuronsPerLayer.length][];
		neuronOutputsErrorSignal = new double[neuronsPerLayer.length][];
		for (int i = 0; i < neuronsPerLayer.length; i++) {
			neuronOutputs[i] = new double[neuronsPerLayer[i]];
			neuronOutputsErrorSignal[i] = new double[neuronsPerLayer[i]];
		}
	}

	// feed forward
	public double[] process(double... networkInput) throws InvalidInputException, NetworkNotInitializedException {
		if (networkInput.length != neuronsPerLayer[0])
			throw new InvalidInputException();
		if (neuronBiases == null || networkWeights == null || neuronOutputs == null || neuronOutputsErrorSignal == null)
			throw new NetworkNotInitializedException();
		// Setting outputs for input layer
		// initially i though that i would need to activate given input so that its
		// between 0 & 1
		// And any input can be give like 5, 10, 1000 for that matter. BUt this is wrong
		// as per the video tutorial https://www.youtube.com/watch?v=aVId8KMsdUU
		// After research, I understood this is what i have to do -
		// this.neuronOutputs[0] = networkInput;
		for (int i = 0; i < neuronsPerLayer[0]; i++) {
			this.neuronOutputs[0][i] = networkInput[i];
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

	public void trainALittle(DataSet trainingSet) throws InvalidInputException, NetworkNotInitializedException {
		Iterator<TrainingExample> iterator = trainingSet.iterator();
		while (iterator.hasNext()) {
			TrainingExample thisExample = iterator.next();
			trainALittle(thisExample.getInput(), thisExample.getOutput());
		}
		log.debug("Completed Training on given training set.");
	}

	// Backpropagate
	private void trainALittle(double[] input, double[] desiredOutput)
			throws InvalidInputException, NetworkNotInitializedException {
		if (desiredOutput.length != neuronsPerLayer[neuronsPerLayer.length - 1])
			throw new InvalidInputException();

		// First feedforward this input forward
		process(input);

		// Calculate the ErrorSignal for all Neurons
		populateSigmoidErrorSignalForAllNeurons(desiredOutput);

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

}
