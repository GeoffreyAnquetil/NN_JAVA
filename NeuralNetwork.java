public class NeuralNetwork {
    private Layer[] layers;

    public NeuralNetwork(int[] architecture) {
        layers = new Layer[architecture.length - 1];
        for (int i = 0; i < architecture.length - 1; i++) {
            layers[i] = new Layer(architecture[i], architecture[i + 1]);
        }
    }

    public double[] predict(double[] inputs) {
        double[] outputs = inputs;
        for (Layer layer : layers) {
            outputs = layer.forward(outputs);
        }
        return outputs;
    }
}
