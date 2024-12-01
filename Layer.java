import java.util.Random;

public class Layer {
    private double[][] weights;
    private double[] biases;
    private double[] outputs;

    public Layer(int inputSize, int outputSize) {
        Random random = new Random();
        // Matrice de poids : w(i,j) poids entre neuronne j de la couche précédente et neuronne i de la couche actuelle
        weights = new double[outputSize][inputSize]; 
        biases = new double[outputSize];

        // Initialisation aléatoire
        for (int i = 0; i < outputSize; i++) {
            biases[i] = random.nextDouble() - 0.5;
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] = random.nextDouble() - 0.5;
            }
        }
    }

    public double[] forward(double[] inputs) {
        outputs = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            double sum = biases[i];
            for (int j = 0; j < inputs.length; j++) {
                sum += weights[i][j] * inputs[j];
            }
            outputs[i] = Util.sigmoid(sum); // Exemple d'activation
        }
        return outputs;
    }

    public double[] getOutputs() {
        return outputs;
    }
}
