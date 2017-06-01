package net.mbretsch.neuronalnet.activators;

/**
 * Most common activation. Uses the logistic function.
 * @author mbretsch
 *
 */
public class SigmoidActivationStrategy implements ActivationStrategy {

	private double c = 1d;

	public SigmoidActivationStrategy(double c) {
		this.c = c;
	}

	@Override
	public double activate(double weightedSum) {

		return 1d / (1d + Math.exp(-c * weightedSum));
	}

	@Override
	public double derivative(double weightedSum) {
		return weightedSum * (1d - weightedSum);
	}

}
