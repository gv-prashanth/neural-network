package com.vadrin.neuralnetwork.models;

import java.util.HashSet;
import java.util.Random;

public class TrainingSet extends HashSet<TrainingExample> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5558379871625775392L;

	public TrainingSet getRandomBatch() {
		// TODO: This needs to be improved.
		TrainingSet toReturn = new TrainingSet();
		while (toReturn.size() <= this.size() / 10) {
			TrainingExample randomExample = this.stream().skip(new Random().nextInt(this.size())).findFirst()
					.orElse(null);
			if (randomExample != null) {
				toReturn.add(randomExample);
			} else {
				break;
			}
		}
		return toReturn;
	}

	public void add(double[] input, double[] output) {
		this.add(new TrainingExample(input, output));
	}
}
