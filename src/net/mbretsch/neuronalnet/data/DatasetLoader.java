package net.mbretsch.neuronalnet.data;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Helper for loading of files that contain input configuration for a neural net.<br>
 * FileFormat:
 * <ul>
 * <li>use # for a commented line which will be ignored</li>
 * <li>every non-empty line is a comma-seperated array of inputs where first number is mapped to first neuron of the inputlayer, second number is mapped to second neuron of inputlayer, etc.</li> 
 * </ul>
 * @author mbretsch
 *
 */
public class DatasetLoader {

	private double[][] inputs;

	public void loadPattern(String filename) {

		InputStream is = DatasetLoader.class.getClassLoader().getResourceAsStream(filename);

		List<String> patterns = new ArrayList<>();

		BufferedReader br = new BufferedReader(new InputStreamReader(is));

		String s;
		try {
			while ((s = br.readLine()) != null) {
				s = s.replaceAll("\\s", "");
				if (s.length() > 0 && s.charAt(0) == '#') {
					continue;
				}
				if (s.indexOf(',') == -1) {
					continue;
				}
				patterns.add(s);
			}
			inputs = new double[patterns.size()][];
			for (int i = 0; i < patterns.size(); i++) {

				String[] input = patterns.get(i).split(",");
				inputs[i] = new double[input.length];
				for (int j = 0; j < input.length; j++) {
					inputs[i][j] = Double.valueOf(input[j]);
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

	public double[][] getInputs() {
		return inputs;
	}

}
