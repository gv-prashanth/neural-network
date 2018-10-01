package com.vadrin.neuralnetwork.models;

public class TrainingSet {

	double[] input;
	double[] output;

	public TrainingSet(double[] input, double[] output) {
		super();
		this.input = input;
		this.output = output;
	}

	public double[] getInput() {
		return input;
	}

	public double[] getOutput() {
		return output;
	}

}
