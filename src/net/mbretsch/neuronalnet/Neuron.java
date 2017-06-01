package net.mbretsch.neuronalnet;

import java.util.ArrayList;
import java.util.List;

import net.mbretsch.neuronalnet.activators.ActivationStrategy;

public class Neuron {

	private List<Synapse> inputSynapses = new ArrayList<Synapse>();
	private List<Synapse> outputSynapses = new ArrayList<Synapse>();
	private ActivationStrategy activationStrategy;
	private double output;
	private double derivative;
	private double weightedSum;
	private double gradient;

	public Neuron(ActivationStrategy activationStrategy) {
		this.activationStrategy = activationStrategy;
		gradient = 0;
	}

	public void addInput(Synapse input) {
		inputSynapses.add(input);
	}

	public void addOutput(Synapse output) {
		outputSynapses.add(output);
	}

	public List<Synapse> getInputs() {
		return inputSynapses;
	}

	public List<Synapse> getOutputs() {
		return outputSynapses;
	}

	public double[] getWeights() {
		double[] weights = new double[inputSynapses.size()];

		for (int i = 0; i < weights.length; i++) {
			weights[i] = inputSynapses.get(i).getWeight();
		}
		return weights;
	}

	private void calcWeightedSum() {
		weightedSum = 0;
		for (int i = 0; i < inputSynapses.size(); i++) {
			weightedSum += inputSynapses.get(i).getWeight() * inputSynapses.get(i).getSourceNeuron().getOutput();
		}
	}

	public void activate() {
		calcWeightedSum();
		if (inputSynapses.size() > 0) {
			this.output = activationStrategy.activate(weightedSum);
		}
		this.derivative = activationStrategy.derivative(output);
	}

	public double getOutput() {
		return this.output;
	}

	public void setOutput(double output) {
		this.output = output;
	}

	public ActivationStrategy getActivationStrategy() {
		return this.activationStrategy;
	}

	public double getDerivative() {
		return this.derivative;
	}

	public double getGradient() {
		return this.gradient;
	}

	public void setGradient(double gradient) {
		this.gradient = gradient;
	}

}
