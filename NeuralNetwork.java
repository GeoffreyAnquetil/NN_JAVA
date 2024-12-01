public class NeuralNetwork {
    private Layer[] layers;

    public NeuralNetwork(int[] architecture) {
        layers = new Layer[architecture.length - 1];
        for (int i = 0; i < architecture.length - 1; i++) {
            layers[i] = new Layer(architecture[i], architecture[i + 1]);
        }
    }

    public double[] feedforward(double[] inputs) {
        double[] activations = inputs;

        // Propagation dans les couches cachées
        for (int i = 0; i < layers.length - 1; i++) {  // Toutes sauf la dernière couche
            activations = layers[i].forward(activations, false);  // false car ce n'est pas la couche de sortie
        }

        // Propagation dans la couche de sortie
        double[] outputLogits = layers[layers.length - 1].forward(activations, true);  // true car c'est la couche de sortie
        
        // Appliquer Softmax à la sortie (pas ReLU)
        return Util.softmax(outputLogits);  // Softmax pour obtenir les probabilités
    }

    public int predict(double[] inputs) {
        // Renvoie la classe prédite (indice du neuronne avec la plus grande probabilité)
        double[] probabilities = feedforward(inputs);
        double maxProbability = 0;
        int predictedClass = 0;
        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > maxProbability) {
                maxProbability = probabilities[i];
                predictedClass = i;
            }
        }
        return predictedClass;
    }

}
