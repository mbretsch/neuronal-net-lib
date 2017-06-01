package net.mbretsch.neuronalnet.backprop;



/**
 * Interface for declaration of Backprop-Algorithms
 * @author mbretsch
 *
 */
public interface BackpropAlgorithm {

	public double useBackprop(double[][] inputs, double[][] expectedOutputs);

	public double useBackpropBatch(double[][] inputs, double[][] expectedOutputs);

	public String getName();

}
