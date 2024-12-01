import java.util.Random;

public class Layer {
    private double[][] weights;
    private double[] biases;
    private double[] outputs;
    private double[][] dWeights; // Gradients des poids
    private double[] dBiases;    // Gradients des biais

    public Layer(int inputSize, int outputSize) {
        Random random = new Random();
        weights = new double[outputSize][inputSize];
        biases = new double[outputSize];
        dWeights = new double[outputSize][inputSize];  // Initialisation des gradients
        dBiases = new double[outputSize];

        for (int i = 0; i < outputSize; i++) {
            biases[i] = random.nextDouble() - 0.5;
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] = random.nextDouble() - 0.5;
            }
        }
    }

    public double[] forward(double[] inputs, boolean isOutputLayer) {
        outputs = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            double sum = biases[i];
            for (int j = 0; j < inputs.length; j++) {
                sum += weights[i][j] * inputs[j];
            }
            // Utilisation de ReLU pour les couches cachées, Softmax sera appliqué en dehors de la couche de sortie
            if (isOutputLayer) {
                outputs[i] = sum; // Pas d'activation dans la couche de sortie ici
            } else {
                outputs[i] = Util.relu(sum); // Activation ReLU pour les couches cachées
            }
        }
        return outputs;
    }

    public void backward(double[] inputs, double[] outputErrors, double learningRate, boolean isOutputLayer) {
        // Calcul des gradients pour les poids et biais
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < inputs.length; j++) {
                if (isOutputLayer) {
                    // Pour la couche de sortie, l'erreur est déjà calculée, donc on utilise directement
                    dWeights[i][j] = outputErrors[i] * inputs[j];
                } else {
                    // Pour les couches cachées, on utilise la dérivée de la fonction d'activation
                    dWeights[i][j] = outputErrors[i] * Util.reluDerivative(inputs[j]) * inputs[j];
                }
            }
            // Calcul des gradients des biais
            dBiases[i] = outputErrors[i];
        }

        // Mise à jour des poids et des biais
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] -= learningRate * dWeights[i][j];
            }
            biases[i] -= learningRate * dBiases[i];
        }
    }

    public double[] getOutputs() {
        return outputs;
    }

    public double[][] getWeights() {
        return weights;
    }
}
