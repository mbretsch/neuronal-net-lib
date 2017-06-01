package net.mbretsch.neuronalnet.config;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

import com.panayotis.gnuplot.JavaPlot;

public class Config {

	public static String PATH_GNUPLOT = "";
	public static JavaPlot JAVA_PLOT = null;

	public static void loadProperties() {
		InputStream is = null;

		Properties properties = new Properties();
		is = Config.class.getClassLoader().getResourceAsStream(Reference.PROPERTIES_FILE);

		try {
			properties.load(is);

			if (properties.containsKey("gnuplot")) {
				PATH_GNUPLOT = properties.getProperty("gnuplot");
				if (JAVA_PLOT == null) {
					JAVA_PLOT = new JavaPlot(PATH_GNUPLOT);
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				is.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	public static void loadProperties(String gnuplotPath) {
		PATH_GNUPLOT = gnuplotPath;
	}

}
