package net.mbretsch.neuronalnet.backprop;

import java.util.List;
import java.util.Random;

import com.panayotis.gnuplot.JavaPlot;
import com.panayotis.gnuplot.plot.AbstractPlot;
import com.panayotis.gnuplot.plot.Axis;
import com.panayotis.gnuplot.plot.DataSetPlot;

import net.mbretsch.neuronalnet.NeuralNetwork;
import net.mbretsch.neuronalnet.config.Config;
import net.mbretsch.neuronalnet.modification.NetModification;

/**
 * Implementation of Backpropagation with three modes (Online-, Batch- and Stochastic-Learning). Uses {@link BackpropAlgorithm} which defines a specific algorithm implementation.
 * @author mbretsch
 *
 */
public class Backprop {

	private NeuralNetwork neuralNetwork;
	private BackpropAlgorithm backpropAlgorithm;

	List<NetModification> netModifications = null;

	public Backprop(NeuralNetwork neuralNetwork, BackpropAlgorithm backpropAlgorithm) {
		this.neuralNetwork = neuralNetwork;
		this.backpropAlgorithm = backpropAlgorithm;
	}

	public Backprop(NeuralNetwork neuralNetwork, BackpropAlgorithm backpropAlgorithm,
			List<NetModification> netModifications) {
		this(neuralNetwork, backpropAlgorithm);
		this.netModifications = netModifications;
	}

	public double onlineLearning(double[][] inputs, double[][] expectedOutputs, int runs) {
		double[][] errors = new double[runs][];
		for (int i = 0; i < runs; i++) {
			errors[i] = new double[] { i, 0 };

			errors[i][1] = this.backpropAlgorithm.useBackprop(inputs, expectedOutputs);
			this.applyModifications();
		}
		addPlot(errors, "online");
		return errors[runs - 1][1];
	}

	public double batchLearning(double[][] inputs, double[][] expectedOutputs, int runs) {
		double[][] errors = new double[runs][];
		for (int i = 0; i < runs; i++) {
			errors[i] = new double[] { i, 0 };

			errors[i][1] = this.backpropAlgorithm.useBackpropBatch(inputs, expectedOutputs);
			this.applyModifications();
		}
		addPlot(errors, "batch");
		return errors[runs - 1][1];
	}

	public double stochasticLearning(double[][] inputs, double[][] expectedOutputs, int runs) {
		Random random = new Random();
		int numPatterns = inputs.length;
		double[][] errors = new double[runs][];

		for (int i = 0; i < runs; i++) {
			double[][] actInputs = new double[numPatterns][];
			double[][] actExpOutputs = new double[numPatterns][];
			for (int j = 0; j < numPatterns; j++) {
				int r = random.nextInt(numPatterns);
				actInputs[j] = inputs[r];
				actExpOutputs[j] = expectedOutputs[r];
			}
			errors[i] = new double[] { i, 0 };
			errors[i][1] = backpropAlgorithm.useBackprop(actInputs, actExpOutputs);
			this.applyModifications();
		}
		this.addPlot(errors, "stochastic");
		return errors[runs - 1][1];
	}

	private void addPlot(double[][] values, String learning) {
		if (Config.JAVA_PLOT != null) {
			AbstractPlot plot = new DataSetPlot(values);
			plot.setTitle(neuralNetwork.getName() + " + " + backpropAlgorithm.getName() + " + " + learning);
			Config.JAVA_PLOT.addPlot(plot);
		}
	}

	public void plot() {
		JavaPlot plot = Config.JAVA_PLOT;
		if (plot != null) {
			Axis runs = plot.getAxis("x");
			runs.setLabel("Run");
			Axis errors = plot.getAxis("y");
			errors.setLabel("Error");
			Config.JAVA_PLOT.plot();
		}
	}

	private void applyModifications() {
		if (netModifications != null) {
			for (NetModification netmod : netModifications) {
				netmod.modifyNet();
			}
		}
	}

	public void printModifications() {
		if (netModifications != null) {
			for (NetModification netmod : netModifications) {
				System.out.println(netmod.toString());
			}
		}
	}

}
