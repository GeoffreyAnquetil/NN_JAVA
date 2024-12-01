import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DatasetLoader {
    
    /**
     * Loads and normalizes vectors from a CSV file.
     * Each line in the file represents a flattened image with pixel values.
     * 
     * @param filePath The path to the CSV file containing the vectors.
     * @return A 2D array where each row represents a normalized image vector.
     * @throws IOException If an error occurs while reading the file.
     */
    public static double[][] loadVectors(String filePath) throws IOException {
        List<double[]> vectors = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                double[] vector = new double[values.length];
                for (int i = 0; i < values.length; i++) {
                    vector[i] = Double.parseDouble(values[i]) / 255.0; // Normalization
                }
                vectors.add(vector);
            }
        }
        return vectors.toArray(new double[0][]);
    }

    /**
     * Loads labels from a CSV file.
     * Each line in the file contains a single integer representing a class label.
     * 
     * @param filePath The path to the CSV file containing the labels.
     * @return An array of integers representing the labels.
     * @throws IOException If an error occurs while reading the file.
     */
    public static int[] loadLabels(String filePath) throws IOException {
        List<Integer> labels = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                labels.add(Integer.parseInt(line.trim()));
            }
        }
        return labels.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * Converts integer labels into one-hot encoded vectors.
     * For each label, a vector of length `numClasses` is created, where only the index
     * corresponding to the label is set to 1.0, and all other indices are set to 0.0.
     * 
     * @param labels The array of integer labels.
     * @param numClasses The total number of classes.
     * @return A 2D array where each row is a one-hot encoded vector for a label.
     */
    public static double[][] oneHotEncodeLabels(int[] labels, int numClasses) {
        double[][] oneHotLabels = new double[labels.length][numClasses];
        for (int i = 0; i < labels.length; i++) {
            oneHotLabels[i][labels[i]] = 1.0;
        }
        return oneHotLabels;
    }
}
