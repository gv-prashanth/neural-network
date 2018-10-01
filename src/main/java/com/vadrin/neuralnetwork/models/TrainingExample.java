package com.vadrin.neuralnetwork.models;

public class TrainingExample {

	double[] input;
	double[] output;

	public TrainingExample(double[] input, double[] output) {
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
