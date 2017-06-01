package net.mbretsch.neuronalnet;

import java.util.ArrayList;
import java.util.List;

public class Layer {

	private List<Neuron> neurons = new ArrayList<Neuron>();
	private Layer previousLayer;
	private Layer nextLayer;
	private Neuron bias;

	public Layer() {
		previousLayer = null;
	}

	public Layer(Layer prevLayer) {
		this.previousLayer = prevLayer;
	}

	public Layer(Layer prevLayer, Neuron bias) {
		this(prevLayer);
		this.bias = bias;
		neurons.add(bias);
	}

	public List<Neuron> getNeurons() {
		return this.neurons;
	}

	public void addNeuron(Neuron neuron) {
		neurons.add(neuron);

		if (previousLayer != null) {
			int countNeurons = previousLayer.getNeurons().size();
			double maxWeight = 1 / Math.pow(countNeurons, 0.5d);
			for (Neuron prevLayerNeuron : previousLayer.getNeurons()) {
				double weight = (Math.random() * 2 * maxWeight) - maxWeight;
				Synapse newSynapse = new Synapse(prevLayerNeuron, neuron, weight);
				neuron.addInput(newSynapse);
				prevLayerNeuron.addOutput(newSynapse);

			}
		}
	}

	public List<Neuron> addRecurentNeurons() {
		if (previousLayer == null) {
			return null;
		}
		List<Neuron> recurentNeurons = new ArrayList<Neuron>();
		for (int i = 0; i < this.neurons.size(); i++) {
			Neuron newNeuron = new Neuron(this.neurons.get(i).getActivationStrategy());
			double maxWeight = 1 / Math.pow(this.neurons.size(), 0.5d);
			for (Neuron neuron : this.neurons) {
				double weight = (Math.random() * 2 * maxWeight) - maxWeight;
				Synapse newSynapse = new Synapse(newNeuron, neuron, weight);
				neuron.addInput(newSynapse);
				newNeuron.addOutput(newSynapse);
			}
			recurentNeurons.add(newNeuron);
			previousLayer.neurons.add(newNeuron);

		}
		return recurentNeurons;
	}

	public void addNeuron(Neuron neuron, double[] weights) {
		neurons.add(neuron);
		if (this.previousLayer != null && this.previousLayer.getNeurons().size() == weights.length) {
			for (int i = 0; i < weights.length; i++) {
				Synapse newSynapse = new Synapse(this.previousLayer.getNeurons().get(i), neuron, weights[i]);
				neuron.addInput(newSynapse);
				this.previousLayer.getNeurons().get(i).addOutput(newSynapse);
			}
		}
	}

	public void feedForward() {

		int biasCount = hasBias() ? 1 : 0;
		for (int i = biasCount; i < neurons.size(); i++) {
			neurons.get(i).activate();
		}
	}

	public Layer getPrevLayer() {
		return this.previousLayer;
	}

	public void setPreviousLayer(Layer prevLayer) {
		this.previousLayer = prevLayer;
	}

	public Layer getNextLayer() {
		return this.nextLayer;
	}

	public void setNextLayer(Layer nextLayer) {
		this.nextLayer = nextLayer;
	}

	public boolean isOutputLayer() {
		return nextLayer == null;
	}

	public boolean hasBias() {
		return bias != null;
	}

}
