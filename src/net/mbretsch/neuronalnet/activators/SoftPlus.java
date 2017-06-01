package net.mbretsch.neuronalnet.activators;

public class SoftPlus implements ActivationStrategy {

	@Override
	public double activate(double weightedSum) {
		return Math.log(1 + Math.exp(weightedSum));
	}

	@Override
	public double derivative(double weightedSum) {
		return 1 / (1 + Math.exp(-weightedSum));
	}

}
