package net.mbretsch.neuronalnet.activators;

/**
 * An ActivationStrategy is used to implement different formulas to activate neurons.
 * @author mbretsch
 *
 */
public interface ActivationStrategy {

	public double activate(double weightedSum);

	public double derivative(double weightedSum);

}
