package net.mbretsch.neuronalnet;

public class Synapse {

	private Neuron sourceNeuron;
	private Neuron targetNeuron;
	private double weight;

	public Synapse(Neuron sourceNeuron, Neuron targetNeuron, double weight) {
		this.sourceNeuron = sourceNeuron;
		this.targetNeuron = targetNeuron;
		this.weight = weight;
	}

	public Neuron getSourceNeuron() {
		return this.sourceNeuron;
	}

	public Neuron getTargetNeuron() {
		return this.targetNeuron;
	}

	public double getWeight() {
		return this.weight;
	}

	public void setWeight(double weight) {
		this.weight = weight;
	}
}
