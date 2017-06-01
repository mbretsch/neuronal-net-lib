package net.mbretsch.neuronalnet.backprop;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import net.mbretsch.neuronalnet.Layer;
import net.mbretsch.neuronalnet.NeuralNetwork;
import net.mbretsch.neuronalnet.Neuron;
import net.mbretsch.neuronalnet.Synapse;

public class BackpropWithMomentum implements BackpropAlgorithm {

	private NeuralNetwork neuralNetwork;
	private double learningRate;
	private double momentum;

	public BackpropWithMomentum(NeuralNetwork neuralNetwork, double learningRate, double momentum) {
		this.neuralNetwork = neuralNetwork;
		this.learningRate = learningRate;
		this.momentum = momentum;
	}

	@Override
	public double useBackprop(double[][] inputs, double[][] expectedOutputs) {
		double error = 0;

		Map<Synapse, Double> neuronDeltaMap = new HashMap<Synapse, Double>();

		for (int i = 0; i < inputs.length; i++) {
			double[] input = inputs[i];
			double[] expectedOutput = expectedOutputs[i];

			List<Layer> layers = neuralNetwork.getLayers();

			neuralNetwork.setInputs(input);
			double[] output = neuralNetwork.getOutput();

			for (int j = layers.size() - 1; j > 0; j--) {
				Layer layer = layers.get(j);

				for (int k = 0; k < layer.getNeurons().size(); k++) {
					Neuron neuron = layer.getNeurons().get(k);
					double neuronError = 0;

					if (layer.isOutputLayer()) {
						neuronError = neuron.getDerivative() * (output[k] - expectedOutput[k]);
					} else {
						neuronError = neuron.getDerivative();

						double sum = 0;
						for (Synapse synapse : neuron.getOutputs()) {
							sum += synapse.getWeight() * synapse.getTargetNeuron().getGradient();
						}
						neuronError *= sum;
					}

					neuron.setGradient(neuronError);
				}
			}

			for (int j = layers.size() - 1; j > 0; j--) {
				Layer layer = layers.get(j);

				for (Neuron neuron : layer.getNeurons()) {

					for (Synapse synapse : neuron.getInputs()) {

						double delta = learningRate * neuron.getGradient() * synapse.getSourceNeuron().getOutput();

						if (neuronDeltaMap.get(synapse) != null) {
							double prevDelta = neuronDeltaMap.get(synapse);
							delta += momentum * prevDelta;
						}

						synapse.setWeight(synapse.getWeight() - delta);
						neuronDeltaMap.put(synapse, delta);
					}
				}
			}

			output = neuralNetwork.getOutput();
			error += error(output, expectedOutput);
		}

		return error;

	}

	@Override
	public double useBackpropBatch(double[][] inputs, double[][] expectedOutputs) {
		double error = 0;

		Map<Synapse, Double> neuronDeltaMap = new HashMap<Synapse, Double>();
		Map<Synapse, Double> weightMap = new HashMap<Synapse, Double>();

		for (int i = 0; i < inputs.length; i++) {
			double[] input = inputs[i];
			double[] expectedOutput = expectedOutputs[i];

			List<Layer> layers = neuralNetwork.getLayers();

			neuralNetwork.setInputs(input);
			double[] output = neuralNetwork.getOutput();

			for (int j = layers.size() - 1; j > 0; j--) {
				Layer layer = layers.get(j);

				for (int k = 0; k < layer.getNeurons().size(); k++) {
					Neuron neuron = layer.getNeurons().get(k);
					double neuronError = 0;

					if (layer.isOutputLayer()) {
						neuronError = neuron.getDerivative() * (output[k] - expectedOutput[k]);
					} else {
						neuronError = neuron.getDerivative();

						double sum = 0;
						for (Synapse synapse : neuron.getOutputs()) {
							sum += synapse.getWeight() * synapse.getTargetNeuron().getGradient();
						}
						neuronError *= sum;
					}

					neuron.setGradient(neuronError);
				}
			}

			for (int j = layers.size() - 1; j > 0; j--) {
				Layer layer = layers.get(j);

				for (Neuron neuron : layer.getNeurons()) {

					for (Synapse synapse : neuron.getInputs()) {

						double delta = learningRate * neuron.getGradient() * synapse.getSourceNeuron().getOutput();

						if (neuronDeltaMap.get(synapse) != null) {
							double prevDelta = neuronDeltaMap.get(synapse);
							delta += momentum * prevDelta;
						}

						neuronDeltaMap.put(synapse, delta);

						if (weightMap.containsKey(synapse)) {
							double weight = weightMap.get(synapse);
							weight -= delta;
							weightMap.put(synapse, weight);
						} else {
							double weight = synapse.getWeight() - delta;
							weightMap.put(synapse, weight);
						}
					}
				}
			}

			output = neuralNetwork.getOutput();
			error += error(output, expectedOutput);
		}
		for (int j = neuralNetwork.getLayers().size() - 1; j > 0; j--) {
			Layer layer = neuralNetwork.getLayers().get(j);
			for (Neuron neuron : layer.getNeurons()) {
				for (Synapse synapse : neuron.getInputs()) {
					synapse.setWeight(weightMap.get(synapse));
				}
			}
		}
		return error;

	}

	public double error(double[] output, double[] expectedOutput) {
		double sum = 0;
		for (int i = 0; i < output.length; i++) {
			sum += Math.pow(expectedOutput[i] - output[i], 2);
		}
		return sum / 2;
	}

	public String getName() {
		String name = String.format("BPMomentum (lRate = %.1f; m = %.2f)", this.learningRate, this.momentum);
		return name;
	}

}
