package net.mbretsch.neuronalnet.data;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

/**
 * Helper to read input configuration from a PNG file. Used for image recognition.
 * @author mbretsch
 *
 */
public class PatternFromPng {

	public double[] getPatternFromPNG(File pngFile) {

		BufferedImage img = null;
		double[] pattern = null;
		try {
			img = ImageIO.read(pngFile);
			int size = img.getHeight() * img.getWidth();
			pattern = new double[size];
			for (int i = 0; i < img.getHeight(); i++) {
				for (int j = 0; j < img.getWidth(); j++) {
					int colPos = img.getRGB(j, i);
					colPos = colPos & 0x0FFFd;
					if (colPos > 0) {
						pattern[i * img.getWidth() + j] = 1f;
					} else {
						pattern[i * img.getWidth() + j] = 0f;
					}
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		return pattern;
	}

}
