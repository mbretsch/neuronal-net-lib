package net.mbretsch.neuronalnet.modification;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import net.mbretsch.neuronalnet.Layer;
import net.mbretsch.neuronalnet.NeuralNetwork;
import net.mbretsch.neuronalnet.Neuron;
import net.mbretsch.neuronalnet.Synapse;

/**
 * Removes synapses between neurons if these synapses have next to none contribution.
 * @author mbretsch
 *
 */
public class ConnectionRemoval implements NetModification {

	private double rate;
	private double removalThreshold;
	private double movingThreshold;
	private NeuralNetwork neuralNetwork;

	private Map<Synapse, Double> prevWeight = new HashMap<Synapse, Double>();
	private Map<Synapse, Double> movingWeight = new HashMap<Synapse, Double>();
	private int countRemovedSynapses = 0;

	public ConnectionRemoval(NeuralNetwork neuralNetwork) {
		this.neuralNetwork = neuralNetwork;
		this.rate = 0.5;
		this.removalThreshold = 0.00005;
		this.movingThreshold = 0.00005;
	}

	@Override
	public void modifyNet() {
		List<Synapse> removalCandidates = new ArrayList<Synapse>();
		for (int i = neuralNetwork.getLayers().size() - 1; i > 0; i--) {
			Layer cLayer = neuralNetwork.getLayers().get(i);
			for (Neuron cNeuron : cLayer.getNeurons()) {
				for (Synapse cSynapse : cNeuron.getInputs()) {
					if (!prevWeight.containsKey(cSynapse)) {
						prevWeight.put(cSynapse, cSynapse.getWeight());
						movingWeight.put(cSynapse, 0d);
					} else {
						double moving = (1 - rate) * movingWeight.get(cSynapse)
								+ rate * Math.abs(prevWeight.get(cSynapse) - cSynapse.getWeight());
						movingWeight.put(cSynapse, moving);
						prevWeight.put(cSynapse, cSynapse.getWeight());
						if (cSynapse.getWeight() < removalThreshold && moving < movingThreshold) {
							removalCandidates.add(cSynapse);
						}
					}
				}
			}
		}
		if (!removalCandidates.isEmpty()) {
			removeSynapses(removalCandidates);
		}

	}

	private void removeSynapses(List<Synapse> synapses) {
		for (int i = synapses.size() - 1; i >= 0; i--) {
			Synapse s = synapses.get(i);
			s.getSourceNeuron().getOutputs().remove(s);
			s.getTargetNeuron().getInputs().remove(s);
			countRemovedSynapses++;
		}
	}

	public String toString() {
		return "Removed " + countRemovedSynapses + " for " + neuralNetwork.getName();
	}

}
