package net.mbretsch.neuronalnet.activators;

public interface ActivationStrategy {

	public double activate(double weightedSum);

	public double derivative(double weightedSum);

}
