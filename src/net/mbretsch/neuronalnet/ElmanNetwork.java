package net.mbretsch.neuronalnet;

import java.util.ArrayList;
import java.util.List;

/**
 * The Implementation for a simple Recurent Net
 * 
 * @author mbretsch
 *
 */
public class ElmanNetwork extends NeuralNetwork {

	protected List<List<Neuron>> recurentLayerNeurons = new ArrayList<List<Neuron>>();

	public ElmanNetwork(String name) {
		super(name);
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

		if (layers.size() > 2) {
			Layer prevLayer = layer.getPrevLayer();
			List<Neuron> recurentNeurons = prevLayer.addRecurentNeurons();
			if (prevLayer == input) {
				input.getNeurons().removeAll(recurentNeurons);
			}
			recurentLayerNeurons.add(recurentNeurons);
			System.out.println("added " + recurentNeurons.size() + " neurons");
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
			for (int i = 1; i < layers.size() - 2; i++) {
				Layer layer = layers.get(i);
				List<Neuron> recurentNeurons = recurentLayerNeurons.get(i - 1);
				List<Neuron> thisRecurentNeurons = recurentLayerNeurons.get(i);
				for (int j = 0; j < layer.getNeurons().size() - 1; j++) {
					Neuron neuron = layer.getNeurons().get(j);
					if (!thisRecurentNeurons.contains(neuron)) {
						recurentNeurons.get(j).setOutput(neuron.getOutput());
					}
				}
			}

		}
	}

	@Override
	public double[] getOutput() {
		for (int i = 0; i < recurentLayerNeurons.get(0).size(); i++) {
			Neuron neuron = recurentLayerNeurons.get(0).get(i);
			neuron.activate();
		}
		return super.getOutput();
	}

}
