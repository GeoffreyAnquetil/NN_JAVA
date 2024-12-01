import java.io.IOException;

public class Main {

    private static final String TEST_LABELS_PATH = "./data/fashion_mnist_test_labels.csv";
    private static final String TEST_VECTORS_PATH = "./data/fashion_mnist_test_vectors.csv";
    private static final String TRAIN_LABELS_PATH = "./data/fashion_mnist_train_labels.csv";
    private static final String TRAIN_VECTORS_PATH = "./data/fashion_mnist_train_vectors.csv";

    public static void main(String[] args) {
        try {
            // Charger les données d'entraînement
            double[][] trainVectors = DatasetLoader.loadVectors(TRAIN_VECTORS_PATH);
            int[] trainLabels = DatasetLoader.loadLabels(TRAIN_LABELS_PATH);
            double[][] oneHotTrainLabels = DatasetLoader.oneHotEncodeLabels(trainLabels, 10);

            // Charger les données de test
            double[][] testVectors = DatasetLoader.loadVectors(TEST_VECTORS_PATH);
            int[] testLabels = DatasetLoader.loadLabels(TEST_LABELS_PATH);
            double[][] oneHotTestLabels = DatasetLoader.oneHotEncodeLabels(testLabels, 10);

            // Créer le réseau de neurones
            NeuralNetwork nn = new NeuralNetwork(new int[] {784, 128, 64, 10});
            // Tester le réseau de neurones sur une image
            double[] prediction = nn.feedforward(trainVectors[0]);
            for (int i = 0; i < prediction.length; i++) {
                System.out.println("Classe " + i + " : " + prediction[i]);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
