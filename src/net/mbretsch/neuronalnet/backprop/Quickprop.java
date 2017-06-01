package net.mbretsch.neuronalnet.backprop;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import net.mbretsch.neuronalnet.Layer;
import net.mbretsch.neuronalnet.NeuralNetwork;
import net.mbretsch.neuronalnet.Neuron;
import net.mbretsch.neuronalnet.Synapse;

public class Quickprop implements BackpropAlgorithm {

	private NeuralNetwork neuralNetwork;
	private double maxWeightGrowFactor;
	private double learningRate;

	private Map<Synapse, Double> lastSynapseDelta = new HashMap<Synapse, Double>();
	private Map<Neuron, Double> lastNeuronError = new HashMap<Neuron, Double>();

	public Quickprop(NeuralNetwork neuralNetwork, double learningRate) {
		this.neuralNetwork = neuralNetwork;
		this.learningRate = learningRate;
		this.maxWeightGrowFactor = 1.75;
	}

	public Quickprop(NeuralNetwork neuralNetwork, double learningRate, double maxWeightGrowFactor) {
		this.neuralNetwork = neuralNetwork;
		this.learningRate = learningRate;
		this.maxWeightGrowFactor = maxWeightGrowFactor;
	}

	@Override
	public double useBackprop(double[][] inputs, double[][] expectedOutputs) {
		double error = 0;

		Map<Neuron, Double> lastNeuronError = new HashMap<Neuron, Double>();
		Map<Synapse, Double> lastSynapseDelta = new HashMap<Synapse, Double>();

		for (int i = 0; i < inputs.length; i++) {
			double[] input = inputs[i];
			double[] expOutput = expectedOutputs[i];

			List<Layer> layers = neuralNetwork.getLayers();
			neuralNetwork.setInputs(input);
			double[] output = neuralNetwork.getOutput();

			for (int j = layers.size() - 1; j > 0; j--) {
				Layer layer = layers.get(j);

				for (int k = 0; k < layer.getNeurons().size(); k++) {
					Neuron neuron = layer.getNeurons().get(k);
					double neuronError = 0;

					if (layer.isOutputLayer()) {
						neuronError = neuron.getDerivative() * (output[k] - expOutput[k]);
					} else {
						neuronError = neuron.getDerivative();
						double sum = 0;
						for (Synapse synapse : neuron.getOutputs()) {
							sum += synapse.getWeight() * synapse.getTargetNeuron().getGradient();
						}
						neuronError *= sum;
					}
					lastNeuronError.put(neuron, neuron.getGradient());
					neuron.setGradient(neuronError);
				}

			}

			for (int j = layers.size() - 1; j > 0; j--) {
				Layer layer = layers.get(j);

				for (Neuron neuron : layer.getNeurons()) {
					for (Synapse synapse : neuron.getInputs()) {
						if (lastSynapseDelta.get(synapse) != null) {
							double delta = lastSynapseDelta.get(synapse) * neuron.getGradient()
									/ (lastNeuronError.get(neuron) - neuron.getGradient());
							if (delta * maxWeightGrowFactor > lastSynapseDelta.get(synapse)) {
								delta = lastSynapseDelta.get(synapse) * maxWeightGrowFactor;
							}
							lastSynapseDelta.put(synapse, delta);
							synapse.setWeight(delta);
						} else {
							double delta = neuron.getGradient() * synapse.getSourceNeuron().getOutput();
							lastSynapseDelta.put(synapse, delta);
							synapse.setWeight(synapse.getWeight() - delta);
						}
					}
				}
			}

			output = neuralNetwork.getOutput();
			error += error(output, expOutput);
		}

		return 0;
	}

	private double error(double[] output, double[] expectedOutput) {
		double sum = 0;
		for (int i = 0; i < output.length; i++) {
			sum += Math.pow(expectedOutput[i] - output[i], 2);
		}
		return sum;
	}

	public String getName() {
		return "QuickProp";
	}

	@Override
	public double useBackpropBatch(double[][] inputs, double[][] expectedOutputs) {
		double error = 0;

		for (int i = 0; i < neuralNetwork.getLayers().size(); i++) {
			Layer layer = neuralNetwork.getLayers().get(i);
			for (int j = 0; j < layer.getNeurons().size(); j++) {
				layer.getNeurons().get(i).setGradient(0d);
			}
		}

		for (int i = 0; i < inputs.length; i++) {
			double[] input = inputs[i];
			double[] expOutput = expectedOutputs[i];

			neuralNetwork.setInputs(input);
			double[] output = neuralNetwork.getOutput();

			List<Layer> layers = neuralNetwork.getLayers();

			for (int j = layers.size() - 1; j > 0; j--) {
				Layer cLayer = layers.get(j);
				for (int k = 0; k < cLayer.getNeurons().size(); k++) {
					Neuron cNeuron = cLayer.getNeurons().get(k);
					double neuronError = cNeuron.getGradient();

					if (cLayer.isOutputLayer()) {
						neuronError += cNeuron.getDerivative() * (output[k] - expOutput[k]);
					} else {
						double sum = 0;
						for (Synapse synapse : cNeuron.getOutputs()) {
							sum += synapse.getWeight() * synapse.getTargetNeuron().getGradient();
						}
						neuronError += cNeuron.getDerivative() * sum;
					}
					cNeuron.setGradient(neuronError);
				}
			}
			error += error(output, expOutput);
		}

		for (int i = neuralNetwork.getLayers().size() - 1; i > 0; i--) {
			Layer cLayer = neuralNetwork.getLayers().get(i);
			int j = 0;
			for (Neuron neuron : cLayer.getNeurons()) {
				if (!lastNeuronError.containsKey(neuron)) {
					for (Synapse synapse : neuron.getInputs()) {
						double delta = neuron.getGradient();
						synapse.setWeight(synapse.getWeight() - delta);
						lastSynapseDelta.put(synapse, delta);
					}
					lastNeuronError.put(neuron, neuron.getGradient());
				} else {
					// System.out.println("Neuron "+i+"
					// "+j+":"+neuron.getError());
					for (Synapse synapse : neuron.getInputs()) {
						if (lastSynapseDelta.get(synapse) > 0) {
							if (neuron.getGradient() > (maxWeightGrowFactor / (1 + maxWeightGrowFactor)
									* lastNeuronError.get(neuron))) {
								lastSynapseDelta.put(synapse, lastSynapseDelta.get(synapse) * maxWeightGrowFactor);
							} else if ((neuron.getGradient() < (maxWeightGrowFactor / (1 + maxWeightGrowFactor)
									* lastNeuronError.get(neuron)))) {
								lastSynapseDelta.put(synapse, lastSynapseDelta.get(synapse) * neuron.getGradient()
										/ (lastNeuronError.get(neuron) - neuron.getGradient()));
							}
							if (neuron.getGradient() > 0) {
								double delta = learningRate * neuron.getGradient() - lastSynapseDelta.get(synapse);
								synapse.setWeight(delta);
								lastSynapseDelta.put(synapse, delta);
							}
						} else if (lastSynapseDelta.get(synapse) < 0) {
							if (neuron.getGradient() < (maxWeightGrowFactor / (1 + maxWeightGrowFactor)
									* lastNeuronError.get(neuron))) {
								lastSynapseDelta.put(synapse, lastSynapseDelta.get(synapse) * maxWeightGrowFactor);
							} else if ((neuron.getGradient() > (maxWeightGrowFactor / (1 + maxWeightGrowFactor)
									* lastNeuronError.get(neuron)))) {
								lastSynapseDelta.put(synapse, lastSynapseDelta.get(synapse) * neuron.getGradient()
										/ (lastNeuronError.get(neuron) - neuron.getGradient()));
							}
							if (neuron.getGradient() > 0) {
								double delta = learningRate * neuron.getGradient() - lastSynapseDelta.get(synapse);
								synapse.setWeight(delta);
								lastSynapseDelta.put(synapse, delta);
							}
						} else if (lastSynapseDelta.get(synapse) == 0) {
							double delta = learningRate * neuron.getGradient();
							synapse.setWeight(delta);
							lastSynapseDelta.put(synapse, delta);
						}

					}
					lastNeuronError.put(neuron, neuron.getGradient());
				}

			}
			j++;
		}

		return error;
	}

}
