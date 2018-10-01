package com.vadrin.neuralnetwork.models;

import java.util.function.DoubleFunction;

public class NetworkConfig {
	private int[] neuronsPerLayer;
	private DoubleFunction<Double> activationFunction;
	private double initialRandomBiasLower;
	private double initialRandomBiasUpper;
	private double initialRandomWeightLower;
	private double initialRandomWeightHigher;
	private double learningRate;

	public NetworkConfig(int[] neuronsPerLayer, DoubleFunction<Double> activationFunction,
			double initialRandomBiasLower, double initialRandomBiasUpper, double initialRandomWeightLower,
			double initialRandomWeightHigher, double learningRate) {
		super();
		this.neuronsPerLayer = neuronsPerLayer;
		this.activationFunction = activationFunction;
		this.initialRandomBiasLower = initialRandomBiasLower;
		this.initialRandomBiasUpper = initialRandomBiasUpper;
		this.initialRandomWeightLower = initialRandomWeightLower;
		this.initialRandomWeightHigher = initialRandomWeightHigher;
		this.learningRate = learningRate;
	}

	public NetworkConfig(int[] neuronsPerLayer, DoubleFunction<Double> activationFunction) {
		this(neuronsPerLayer, activationFunction, 0.3d, 0.7d, -0.5d, 0.7d, 0.3d);
	}

	public NetworkConfig(DoubleFunction<Double> activationFunction, int... neuronsPerLayer) {
		this(neuronsPerLayer, activationFunction);
	}

	public int[] getNeuronsPerLayer() {
		return neuronsPerLayer;
	}

	public DoubleFunction<Double> getActivationFunction() {
		return activationFunction;
	}

	public double getInitialRandomBiasLower() {
		return initialRandomBiasLower;
	}

	public double getInitialRandomBiasUpper() {
		return initialRandomBiasUpper;
	}

	public double getInitialRandomWeightLower() {
		return initialRandomWeightLower;
	}

	public double getInitialRandomWeightHigher() {
		return initialRandomWeightHigher;
	}

	public double getLearningRate() {
		return learningRate;
	}

}
