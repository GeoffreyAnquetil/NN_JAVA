import java.util.Random;

public class NeuralNetwork {

    // Hyperparamètres
    private static final double LEARNING_RATE = 0.01;
    private static final int INPUT_SIZE = 784;  // Par exemple pour Fashion MNIST
    private static final int HIDDEN_SIZE = 128; // Taille de la couche cachée
    private static final int OUTPUT_SIZE = 10;  // Nombre de classes (Fashion MNIST)
    
    // Poids et biais
    private double[][] weightsInputHidden;
    private double[] biasHidden;
    private double[][] weightsHiddenOutput;
    private double[] biasOutput;
    
    // Constructeur
    public NeuralNetwork() {
        Random rand = new Random();
        
        // Initialisation des poids et des biais
        weightsInputHidden = new double[INPUT_SIZE][HIDDEN_SIZE];
        biasHidden = new double[HIDDEN_SIZE];
        weightsHiddenOutput = new double[HIDDEN_SIZE][OUTPUT_SIZE];
        biasOutput = new double[OUTPUT_SIZE];
        
        for (int i = 0; i < INPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                weightsInputHidden[i][j] = rand.nextGaussian() * 0.1; // Initialisation aléatoire
            }
        }
        
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            biasHidden[i] = rand.nextGaussian() * 0.1;
        }
        
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                weightsHiddenOutput[i][j] = rand.nextGaussian() * 0.1;
            }
        }
        
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            biasOutput[i] = rand.nextGaussian() * 0.1;
        }
    }
    
    // Fonction d'activation sigmoïde
    public double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    // Dérivée de la fonction sigmoïde
    public double sigmoidDerivative(double x) {
        return x * (1 - x);
    }
    
    // Propagation avant
    public double[] feedForward(double[] input) {
        // Calcul de la sortie de la couche cachée
        double[] hidden = new double[HIDDEN_SIZE];
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            hidden[i] = 0;
            for (int j = 0; j < INPUT_SIZE; j++) {
                hidden[i] += input[j] * weightsInputHidden[j][i];
            }
            hidden[i] += biasHidden[i];
            hidden[i] = sigmoid(hidden[i]);
        }
        
        // Calcul de la sortie finale
        double[] output = new double[OUTPUT_SIZE];
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            output[i] = 0;
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                output[i] += hidden[j] * weightsHiddenOutput[j][i];
            }
            output[i] += biasOutput[i];
            output[i] = sigmoid(output[i]);
        }
        
        return output;
    }
    
    // Fonction de perte (erreur quadratique)
    public double calculateLoss(double[] output, double[] target) {
        double loss = 0;
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            loss += 0.5 * Math.pow(target[i] - output[i], 2);
        }
        return loss;
    }
    
    // Rétropropagation et mise à jour des poids
    public void backpropagate(double[] input, double[] target) {
        // Propagation avant
        double[] hidden = new double[HIDDEN_SIZE];
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            hidden[i] = 0;
            for (int j = 0; j < INPUT_SIZE; j++) {
                hidden[i] += input[j] * weightsInputHidden[j][i];
            }
            hidden[i] += biasHidden[i];
            hidden[i] = sigmoid(hidden[i]);
        }
        
        double[] output = new double[OUTPUT_SIZE];
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            output[i] = 0;
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                output[i] += hidden[j] * weightsHiddenOutput[j][i];
            }
            output[i] += biasOutput[i];
            output[i] = sigmoid(output[i]);
        }
        
        // Calcul des erreurs (couche de sortie)
        double[] outputErrors = new double[OUTPUT_SIZE];
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            outputErrors[i] = (target[i] - output[i]) * sigmoidDerivative(output[i]);
        }
        
        // Calcul des erreurs (couche cachée)
        double[] hiddenErrors = new double[HIDDEN_SIZE];
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            hiddenErrors[i] = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                hiddenErrors[i] += outputErrors[j] * weightsHiddenOutput[i][j];
            }
            hiddenErrors[i] *= sigmoidDerivative(hidden[i]);
        }
        
        // Mise à jour des poids et des biais (couche cachée -> sortie)
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                weightsHiddenOutput[i][j] += LEARNING_RATE * outputErrors[j] * hidden[i];
            }
        }
        
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            biasOutput[i] += LEARNING_RATE * outputErrors[i];
        }
        
        // Mise à jour des poids et des biais (entrée -> cachée)
        for (int i = 0; i < INPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                weightsInputHidden[i][j] += LEARNING_RATE * hiddenErrors[j] * input[i];
            }
        }
        
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            biasHidden[i] += LEARNING_RATE * hiddenErrors[i];
        }
    }
    
    // Entraînement du modèle
    public void train(double[][] inputs, double[][] targets, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;
            for (int i = 0; i < inputs.length; i++) {
                double[] output = feedForward(inputs[i]);
                totalLoss += calculateLoss(output, targets[i]);
                backpropagate(inputs[i], targets[i]);
            }
            System.out.println("Epoch " + epoch + " - Loss: " + totalLoss);
        }
    }

    // Exemple d'utilisation
    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork();
        
        // Exemple de données (inputs et cibles)
        double[][] inputs = new double[][] {
            // Exemple avec deux entrées pour un problème simple
            {0.0, 0.0},
            {0.0, 1.0},
            {1.0, 0.0},
            {1.0, 1.0}
        };
        
        double[][] targets = new double[][] {
            {0.0},
            {1.0},
            {1.0},
            {0.0}
        };
        
        nn.train(inputs, targets, 1000);  // Entraîner pendant 1000 époques
    }
}
