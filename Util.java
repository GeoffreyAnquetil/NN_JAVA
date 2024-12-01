public class Util {
    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public static double sigmoidDerivative(double x) {
        return x * (1 - x);
    }

    public static double relu(double x) {
        return Math.max(0, x);  // ReLU : f(x) = max(0, x)
    }

    public static double reluDerivative(double x) {
        if (x > 0) {
            return 1.0;
        } else {
            return 0.0;
        }
    }

    public static double[] softmax(double[] logits) {
        double sum = 0;
        for (double logit : logits) {
            sum += Math.exp(logit);  // Somme des exponentielles des scores
        }

        double[] probabilities = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probabilities[i] = Math.exp(logits[i]) / sum;  // Application de Softmax
        }
        return probabilities;
    }

    public static int argmax(double[] array) {
        int maxIdx = 0;
        double maxVal = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxVal) {
                maxVal = array[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

}   
