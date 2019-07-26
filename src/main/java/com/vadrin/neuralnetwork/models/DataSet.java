package com.vadrin.neuralnetwork.models;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

//TODO: This name needs to be changed. And We are using dataset for both training and test sets.. 
//But the class extends TrainingExample class which is confusing. Needs to be fixed.
public class DataSet extends HashSet<TrainingExample> {

	private static final long serialVersionUID = 5558379871625775392L;

	// TODO: Needs cleanup
	public DataSet getRandomSet(double sizeFactor) {
		List<TrainingExample> list = new ArrayList<TrainingExample>(this);
		Collections.shuffle(list);
		Set<TrainingExample> randomSet = new HashSet<TrainingExample>(
				list.subList(0, (int) (list.size() * sizeFactor)));
		DataSet toReturn = new DataSet();
		toReturn.addAll(randomSet);
		return toReturn;
		// return this;
	}

	public DataSet get() {
		return this;
	}

	public void add(double[] input, double[] output) {
		this.add(new TrainingExample(input, output));
	}
}
