import java.io.IOException;

public class Main {

    private static final String TEST_LABELS_PATH = "./data/fashion_mnist_test_labels.csv";
    private static final String TEST_VECTORS_PATH = "./data/fashion_mnist_test_vectors.csv";
    private static final String TRAIN_LABELS_PATH = "./data/fashion_mnist_train_labels.csv";
    private static final String TRAIN_VECTORS_PATH = "./data/fashion_mnist_train_vectors.csv";

    // Exemple d'utilisation
    public static void main(String[] args) {

        // On mesure le temps total d'exécution
        long startTime = System.currentTimeMillis();

        NeuralNetwork nn = new NeuralNetwork();
        try {
            double[][] trainVectors = DatasetLoader.loadVectors(TRAIN_VECTORS_PATH);
            int[] trainLabels = DatasetLoader.loadLabels(TRAIN_LABELS_PATH);
            double[][] oneHotTrainLabels = DatasetLoader.oneHotEncodeLabels(trainLabels, 10);

            double[][] testVectors = DatasetLoader.loadVectors(TEST_VECTORS_PATH);
            int[] testLabels = DatasetLoader.loadLabels(TEST_LABELS_PATH);
            double[][] oneHotTestLabels = DatasetLoader.oneHotEncodeLabels(testLabels, 10);

            // Entraînement
            nn.train(trainVectors, oneHotTrainLabels, 30);
            // On évalue le modèle
            double accuracy = 0;
            for (int i = 0; i < testVectors.length; i++) {
                double[] output = nn.feedForward(testVectors[i]);
                int predictedLabel = Util.argmax(output);
                if (predictedLabel == testLabels[i]) {
                    accuracy++;
                }
            }
            accuracy /= testVectors.length;
            System.out.println("Accuracy: " + accuracy);

            long endTime = System.currentTimeMillis();
            System.out.println("Execution time: " + (endTime - startTime)/1000 + "s");

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
