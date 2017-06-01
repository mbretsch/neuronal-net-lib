package net.mbretsch.neuronalnet.kohonen;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.panayotis.gnuplot.JavaPlot;
import com.panayotis.gnuplot.dataset.GenericDataSet;
import com.panayotis.gnuplot.dataset.Point;
import com.panayotis.gnuplot.dataset.PointDataSet;
import com.panayotis.gnuplot.plot.DataSetPlot;

import net.mbretsch.neuronalnet.Neuron;
import net.mbretsch.neuronalnet.Synapse;
import net.mbretsch.neuronalnet.activators.ActivationStrategy;
import net.mbretsch.neuronalnet.activators.SigmoidActivationStrategy;
import net.mbretsch.neuronalnet.config.Config;
import net.mbretsch.neuronalnet.javaplot.parser.LabeledParser;

/**
 * Implementation of Kohonen-Maps/Self Organazing Maps.
 * @author mbretsch
 *
 */
public class Kohonen {

	private List<Neuron> inputNeurons = new ArrayList<Neuron>();
	private List<List<Neuron>> map = new ArrayList<List<Neuron>>();
	private int countInputs;
	private int dimension;
	private double learningRate = 0.1d;

	private boolean useEuclidean = false;

	private int winnerX = 0;
	private int winnerY = 0;

	private GenericDataSet winners = new GenericDataSet(new LabeledParser());

	public Kohonen(int countInputs, int mapDimension) {
		this.countInputs = countInputs;
		this.dimension = mapDimension;
		this.createMap();

	}

	public Kohonen(int countInputs, int mapDimension, double learningRate, boolean useEuclidean) {
		this(countInputs, mapDimension);
		this.useEuclidean = useEuclidean;
		this.learningRate = learningRate;
	}

	public void setInputs(double[] input) {
		for (int i = 0; i < input.length; i++) {
			inputNeurons.get(i).setOutput(input[i]);
		}
	}

	public void step() {
		this.calcArousal();
		this.getWinner();
		String label = "(";
		for (int i = 0; i < inputNeurons.size(); i++) {
			label += String.format("%.0f", inputNeurons.get(i).getOutput());
			if (i < inputNeurons.size() - 1) {
				label += ",";
			}
		}
		label += ")";
		ArrayList<String> winnerEntry = new ArrayList<>();
		winnerEntry.add(String.valueOf(winnerX));
		winnerEntry.add(String.valueOf(winnerY));
		winnerEntry.add(label);
		winners.add(winnerEntry);
	}

	public void trainMap(double[][] inputs, int countRuns) {
		Random rand = new Random();
		double radius = this.dimension / 2;
		double radiusDecay = countRuns / Math.log(radius);

		for (int i = 0; i < countRuns; i++) {
			double currentRadius = radius * Math.exp((-i / radiusDecay));
			double currentLearningRate = this.learningRate * Math.exp(-i / countRuns);
			int n = rand.nextInt(inputs.length);
			this.setInputs(inputs[n]);
			this.calcArousal();
			this.getWinner();
			for (int j = 0; j < map.size(); j++) {
				List<Neuron> row = map.get(j);
				for (int k = 0; k < row.size(); k++) {
					Neuron neuron = row.get(k);
					double distSq = (winnerX - k) * (winnerX - k) + (winnerY - j) * (winnerY - j);
					if (distSq < currentRadius * currentRadius) {
						double currentInfluence = Math.exp(-distSq / (2 * currentRadius * currentRadius));
						for (int l = 0; l < neuron.getInputs().size(); l++) {
							Synapse synapse = neuron.getInputs().get(l);
							double weight = synapse.getWeight() + currentInfluence * currentLearningRate
									* (synapse.getSourceNeuron().getOutput() - synapse.getWeight());
							synapse.setWeight(weight);
						}
					}
				}
			}
		}

	}

	public void setEuclidean(boolean useEuclidean) {
		this.useEuclidean = useEuclidean;
	}

	private void createMap() {
		JavaPlot plot = Config.JAVA_PLOT;
		PointDataSet<Integer> dataSet = new PointDataSet<Integer>();
		// GenericDataSet dataSet = new GenericDataSet(new LabeledParser());

		ActivationStrategy as = new SigmoidActivationStrategy(2.0d);
		for (int i = 0; i < this.countInputs; i++) {
			inputNeurons.add(new Neuron(as));
		}

		for (int i = 0; i < this.dimension; i++) {
			List<Neuron> row = new ArrayList<Neuron>();
			for (int j = 0; j < this.dimension; j++) {
				Neuron newNeuron = new Neuron(as);
				for (int k = 0; k < inputNeurons.size(); k++) {
					double weight = Math.random();
					Synapse newSynapse = new Synapse(inputNeurons.get(k), newNeuron, weight);
					newNeuron.addInput(newSynapse);
					inputNeurons.get(k).addOutput(newSynapse);
				}
				row.add(newNeuron);
				dataSet.add(new Point<Integer>(j, i));
			}
			map.add(row);
		}
		if (plot != null) {
			plot.set("key", "rmargin");
			DataSetPlot dsp = new DataSetPlot(dataSet);
			dsp.setTitle("Map");
			// dsp.set("using 1:2:(sprintf(\"%s\", $3)) with labels point pt 7
			// offset char 1,1");
			// dsp.set("using 1:2:(sprintf(\"%s\",stringcolumn(3))) with labels
			// ");
			plot.addPlot(dsp);

		}
	}

	private void calcArousal() {
		for (List<Neuron> row : map) {
			for (Neuron neuron : row) {
				neuron.activate();
			}
		}
	}

	private Neuron getWinner() {
		Neuron winner = map.get(0).get(0);
		if (this.useEuclidean) {
			double bestSum = countInputs * countInputs;
			for (int i = 0; i < map.size(); i++) {
				List<Neuron> row = map.get(i);
				for (int j = 0; j < row.size(); j++) {
					Neuron neuron = row.get(j);
					double sum = 0;
					for (int k = 0; k < neuron.getInputs().size(); k++) {
						Synapse synapse = neuron.getInputs().get(k);
						sum += Math.pow((synapse.getSourceNeuron().getOutput() - synapse.getWeight()), 2);
					}
					if (sum < bestSum) {
						winnerX = j;
						winnerY = i;
						winner = neuron;
						bestSum = sum;
					}
				}
			}
		} else {
			for (int i = 0; i < map.size(); i++) {
				List<Neuron> row = map.get(i);
				for (int j = 0; j < row.size(); j++) {
					Neuron neuron = row.get(j);
					if (neuron.getOutput() > winner.getOutput()) {
						winner = neuron;
						winnerX = j;
						winnerY = i;
					}
				}
			}
		}
		return winner;
	}

	public void plot() {
		JavaPlot plot = Config.JAVA_PLOT;

		if (plot != null) {
			DataSetPlot dsp = new DataSetPlot(winners);
			dsp.set("using 1:2:(sprintf(\"%s\", stringcolumn(3))) with labels point pt 7 offset char 1,1");
			dsp.setTitle("Winners");
			plot.addPlot(dsp);
			plot.plot();
		}
	}

}
