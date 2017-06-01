package net.mbretsch.neuronalnet;

import java.util.ArrayList;
import java.util.List;

/**
 * A NeuralNetwork is used to Configure a Neural Net, setting Inputs and getting outputs after calculating.
 * @author mbretsch
 *
 */
public class NeuralNetwork {

	protected String name;
	protected Layer input;
	protected List<Layer> layers = new ArrayList<Layer>();
	protected Layer output;

	public NeuralNetwork(String name) {
		this.name = name;
	}

	public void addLayer(Layer layer) {
		layers.add(layer);

		if (layers.size() == 1) {
			this.input = layer;
		}

		if (layers.size() > 1) {
			Layer prevLayer = layers.get(layers.size() - 2);
			prevLayer.setNextLayer(layer);
		}

		output = layer;
	}

	public void setInputs(double[] inputs) {
		if (input != null) {

			int biasCount = input.hasBias() ? 1 : 0;
			if (input.getNeurons().size() - biasCount == inputs.length) {
				List<Neuron> neurons = input.getNeurons();
				for (int i = biasCount; i < neurons.size(); i++) {
					neurons.get(i).setOutput(inputs[i - biasCount]);
				}
			}
		}
	}

	public String getName() {
		return this.name;
	}

	public double[] getOutput() {
		double[] outputs = new double[output.getNeurons().size()];

		for (int i = 1; i < layers.size(); i++) {
			layers.get(i).feedForward();
		}

		for (int i = 0; i < output.getNeurons().size(); i++) {
			outputs[i] = output.getNeurons().get(i).getOutput();
		}

		return outputs;
	}

	public List<Layer> getLayers() {
		return this.layers;
	}

	public Layer getOutputLayer() {
		return layers.get(layers.size() - 1);
	}

	public void reset() {
		for (Layer layer : layers) {
			for (Neuron neuron : layer.getNeurons()) {
				int countSynapse = neuron.getInputs().size();
				double maxWeight = 1 / Math.pow(countSynapse, 0.5d);
				for (Synapse synapse : neuron.getInputs()) {
					double weight = (Math.random() * 2 * maxWeight) - maxWeight;
					synapse.setWeight(weight);
				}
			}
		}
	}

}
