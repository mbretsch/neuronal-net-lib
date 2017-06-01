package net.mbretsch.neuronalnet.activators;

public class ThresholdActivationStrategy implements ActivationStrategy {

	private double threshold;

	public ThresholdActivationStrategy(double threshold) {
		this.threshold = threshold;
	}

	@Override
	public double activate(double weightedSum) {
		return (weightedSum > this.threshold ? 1 : 0);
	}

	@Override
	public double derivative(double weightedSum) {
		return 0;
	}

}
