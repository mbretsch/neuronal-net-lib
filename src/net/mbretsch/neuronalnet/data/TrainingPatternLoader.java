package net.mbretsch.neuronalnet.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;
/**
 * Helper for loading of files that contain a training set for the training phase of a neural net.<br>
 * FileFormat:
 * <ul>
 * <li>use # for a commented line which will be ignored</li>
 * <li>every non-empty line is a comma-seperated array of inputs where first number is mapped to first neuron of the inputlayer, second number is mapped to second neuron of inputlayer, etc.</li> 
 * <li>outputs are mapped in the same way after a pipe character '|'</li>
 * </ul>
 * @author mbretsch
 *
 */

public class TrainingPatternLoader {

	private double[][] inputs;
	private double[][] outputs;

	public void loadPattern(String filename) {

		InputStream is = TrainingPatternLoader.class.getClassLoader().getResourceAsStream(filename);

		List<String> patterns = new ArrayList<>();

		BufferedReader br = new BufferedReader(new InputStreamReader(is));

		String s;
		try {
			while ((s = br.readLine()) != null) {
				s = s.replaceAll("\\s", "");
				if (s.length() > 0 && s.charAt(0) == '#') {
					continue;
				}
				if (s.indexOf('|') == -1) {
					continue;
				}
				patterns.add(s);
			}
			inputs = new double[patterns.size()][];
			outputs = new double[patterns.size()][];
			for (int i = 0; i < patterns.size(); i++) {

				String[] patternSplit = patterns.get(i).split("\\|");
				if (patternSplit.length < 2) {
					continue;
				}
				String[] input = patternSplit[0].split(",");
				String[] output = patternSplit[1].split(",");
				inputs[i] = new double[input.length];
				outputs[i] = new double[output.length];
				for (int j = 0; j < input.length; j++) {
					inputs[i][j] = Double.valueOf(input[j]);
				}
				for (int j = 0; j < output.length; j++) {
					outputs[i][j] = Double.valueOf(output[j]);
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				br.close();
				is.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

	}

	public void loadPatternsFromPNG(String dirname) {
		PatternFromPng pfp = new PatternFromPng();

		try {
			File directory = new File(TrainingPatternLoader.class.getClassLoader().getResource(dirname).toURI());
			File[] files = directory.listFiles();
			inputs = new double[files.length][];
			outputs = new double[files.length][];
			for (int i = 0; i < files.length; i++) {
				inputs[i] = pfp.getPatternFromPNG(files[i]);
				outputs[i] = new double[files.length];
				outputs[i][i] = 1;
			}
		} catch (URISyntaxException e) {
			e.printStackTrace();
		}
	}

	public double[][] getInputs() {
		return inputs;
	}

	public double[][] getOutputs() {
		return outputs;
	}

}
