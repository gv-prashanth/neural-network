package com.vadrin.neuralnetwork.commons.utils;

import com.vadrin.neuralnetwork.commons.exceptions.InvalidInputException;

public class ArrayUtils {

	public static double[] createRandomArray(int size, double lower_bound, double upper_bound) throws InvalidInputException {
		if (size < 1) {
			throw new InvalidInputException();
		}
		double[] ar = new double[size];
		for (int i = 0; i < size; i++) {
			ar[i] = randomValue(lower_bound, upper_bound);
		}
		return ar;
	}

	public static double[][] createRandomArray(int sizeX, int sizeY, double lower_bound, double upper_bound) throws InvalidInputException {
		if (sizeX < 1 || sizeY < 1) {
			throw new InvalidInputException();
		}
		double[][] ar = new double[sizeX][sizeY];
		for (int i = 0; i < sizeX; i++) {
			ar[i] = createRandomArray(sizeY, lower_bound, upper_bound);
		}
		return ar;
	}

	public static double randomValue(double lower_bound, double upper_bound) {
		return Math.random() * (upper_bound - lower_bound) + lower_bound;
	}

}
