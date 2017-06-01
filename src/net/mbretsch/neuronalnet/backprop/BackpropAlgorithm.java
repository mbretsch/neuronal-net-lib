package net.mbretsch.neuronalnet.backprop;

public interface BackpropAlgorithm {

	public double useBackprop(double[][] inputs, double[][] expectedOutputs);

	public double useBackpropBatch(double[][] inputs, double[][] expectedOutputs);

	public String getName();

}
