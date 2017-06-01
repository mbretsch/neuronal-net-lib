package net.mbretsch.neuronalnet.javaplot.parser;

import com.panayotis.gnuplot.dataset.parser.DataParser;

/**
 * Helper for JavaPlot to parse labeled GnuPlot datasets.
 * @author mbretsch
 *
 */
public class LabeledParser implements DataParser {

	@Override
	public boolean isValid(String data, int index) {
		if (index == 0 || index == 1) {
			try {
				Double.parseDouble(data);
			} catch (Exception e) {
				return false;
			}
		}
		return true;
	}

}
