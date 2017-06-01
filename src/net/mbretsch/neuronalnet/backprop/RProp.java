package net.mbretsch.neuronalnet.backprop;

import java.util.HashMap;
import java.util.Map;

import net.mbretsch.neuronalnet.Layer;
import net.mbretsch.neuronalnet.NeuralNetwork;
import net.mbretsch.neuronalnet.Neuron;
import net.mbretsch.neuronalnet.Synapse;

/**
 * RProp implementation.
 * @author mbretsch
 *
 */
public class RProp implements BackpropAlgorithm {

	private NeuralNetwork neuralNetwork;
	private double nplus;
	private double nminus;
	private double learningRate;

	private Map<Neuron, Double> lastGradient = new HashMap<Neuron, Double>();
	private Map<Synapse, Double> weightMap = new HashMap<Synapse, Double>();

	public RProp(NeuralNetwork neuralNetwork) {
		this.neuralNetwork = neuralNetwork;
		this.nplus = 1.2;
		this.nminus = 0.5;
		this.learningRate = 0.6d;
	}

	@Override
	public double useBackprop(double[][] inputs, double[][] expectedOutputs) {
		double error = 0;
		for (int i = 0; i < inputs.length; i++) {
			double[] input = inputs[i];
			double[] expOutput = expectedOutputs[i];

			neuralNetwork.setInputs(input);
			double[] output = neuralNetwork.getOutput();
			error += error(output, expOutput);

			for (int j = neuralNetwork.getLayers().size() - 1; j > 0; j--) {
				Layer cLayer = neuralNetwork.getLayers().get(j);
				for (int k = 0; k < cLayer.getNeurons().size(); k++) {
					Neuron cNeuron = cLayer.getNeurons().get(k);
					double gradient = 0;

					if (cLayer.isOutputLayer()) {
						gradient = cNeuron.getDerivative() * (output[k] - expOutput[k]);
					} else {
						gradient = cNeuron.getDerivative();
						double sum = 0;

						for (Synapse synapse : cNeuron.getOutputs()) {
							sum += synapse.getWeight() * synapse.getTargetNeuron().getGradient();
						}
						gradient *= sum;
					}
					cNeuron.setGradient(gradient);
				}
			}

			for (int j = neuralNetwork.getLayers().size() - 1; j > 0; j--) {
				Layer cLayer = neuralNetwork.getLayers().get(j);
				for (Neuron cNeuron : cLayer.getNeurons()) {
					for (Synapse cSynapse : cNeuron.getInputs()) {
						double delta = 0;
						if (!lastGradient.containsKey(cNeuron)) {
							delta = learningRate * cNeuron.getGradient() * cSynapse.getSourceNeuron().getOutput();
							cSynapse.setWeight(cSynapse.getWeight() - delta);
						} else {
							delta = learningRate * cNeuron.getGradient() * cSynapse.getSourceNeuron().getOutput();
							if (lastGradient.get(cNeuron) * cNeuron.getGradient() > 0) {
								delta *= nplus;
							} else if (lastGradient.get(cNeuron) * cNeuron.getGradient() < 0) {
								delta *= nminus;
							}
							cSynapse.setWeight(cSynapse.getWeight() - delta);
						}
					}
					lastGradient.put(cNeuron, cNeuron.getGradient());
				}
			}

		}

		return error;
	}

	@Override
	public double useBackpropBatch(double[][] inputs, double[][] expectedOutputs) {
		double error = 0;

		for (int i = 0; i < inputs.length; i++) {
			double[] input = inputs[i];
			double[] expOutput = expectedOutputs[i];

			neuralNetwork.setInputs(input);
			double[] output = neuralNetwork.getOutput();
			error += error(output, expOutput);

			for (int j = neuralNetwork.getLayers().size() - 1; j > 0; j--) {
				Layer cLayer = neuralNetwork.getLayers().get(j);
				for (int k = 0; k < cLayer.getNeurons().size(); k++) {
					Neuron cNeuron = cLayer.getNeurons().get(k);
					double gradient = 0;

					if (cLayer.isOutputLayer()) {
						gradient = cNeuron.getDerivative() * (output[k] - expOutput[k]);
					} else {
						gradient = cNeuron.getDerivative();
						double sum = 0;

						for (Synapse synapse : cNeuron.getOutputs()) {
							sum += synapse.getWeight() * synapse.getTargetNeuron().getGradient();
						}
						gradient *= sum;
					}
					cNeuron.setGradient(gradient);
				}
			}

			for (int j = neuralNetwork.getLayers().size() - 1; j > 0; j--) {
				Layer cLayer = neuralNetwork.getLayers().get(j);
				for (Neuron cNeuron : cLayer.getNeurons()) {
					for (Synapse cSynapse : cNeuron.getInputs()) {
						double delta = 0;
						if (!weightMap.containsKey(cSynapse)) {
							delta = learningRate * cNeuron.getGradient() * cSynapse.getSourceNeuron().getOutput();
							weightMap.put(cSynapse, (cSynapse.getWeight() - delta));
						} else {
							delta = learningRate * cNeuron.getGradient() * cSynapse.getSourceNeuron().getOutput();
							if (lastGradient.get(cNeuron) * cNeuron.getGradient() > 0) {
								delta *= nplus;
							} else if (lastGradient.get(cNeuron) * cNeuron.getGradient() < 0) {
								delta *= nminus;
							}
							double weight = weightMap.get(cSynapse);
							weight -= delta;
							weightMap.put(cSynapse, weight);
						}
					}
					lastGradient.put(cNeuron, cNeuron.getGradient());
				}
			}

		}

		for (int i = neuralNetwork.getLayers().size() - 1; i > 0; i--) {
			Layer cLayer = neuralNetwork.getLayers().get(i);
			for (Neuron cNeuron : cLayer.getNeurons()) {
				for (Synapse cSynapse : cNeuron.getInputs()) {
					cSynapse.setWeight(weightMap.get(cSynapse));
				}
			}
		}

		return error;
	}

	@Override
	public String getName() {
		return String.format("RProp (n+ = %.2f; n- = %.2f)", nplus, nminus);
	}

	private double error(double[] output, double[] expOutput) {
		double sum = 0;
		for (int i = 0; i < output.length; i++) {
			sum += Math.pow((expOutput[i] - output[i]), 2);
		}
		return sum / 2;

	}

}
