package net.mbretsch.neuronalnet.modification;

/**
 * Interface for network manipulation. NetModifcation are stored in an array and applied after certain conditions are met during learning.
 * @author mbretsch
 *
 */
public interface NetModification {

	public void modifyNet();

}
