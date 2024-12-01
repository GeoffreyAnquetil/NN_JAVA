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

            // Affichage pour vérification
            System.out.println("Train Vectors: " + trainVectors.length);
            System.out.println("Train Labels: " + trainLabels.length);
            System.out.println("Test Vectors: " + testVectors.length);
            System.out.println("Test Labels: " + testLabels.length);
            System.out.print("\n");
            System.out.println("Train Vectors exemple: " + trainVectors[110][4]);
            System.out.println("Train Labels exemple: " + trainLabels[108]);
            System.out.println("One Hot Train Labels exemple: " + DatasetLoader.oneHotEncodeLabels(trainLabels, 10)[108][6]);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
