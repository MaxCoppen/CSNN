//-----------------------------------------------------------------------
// <copyright file="NeuralNetwork.cs" company="RX">
//     Author: Max Coppen
//     Copyright (c) Developer Express. All rights reserved.
// </copyright>
//-----------------------------------------------------------------------

using System;
using System.Linq;

namespace CSNN
{
    public class Neuron
    {
        // activation of this neuron.
        public float activation;
        // bias of this neuron.
        public float bias;
        // weights leading into this neuron.
        public float[] weights;
        // the previous layer.
        public Neuron[] input;

        /// <summary>
        /// Calculates the activation of this neuron and assigns it.
        /// </summary>
        public void CalculateActivation()
        {
            float new_activation = 0;

            // Loop over the input neurons.
            for (int k = 0; k < input.Length; k++)
            {
                // Weight from input neuron to (this).
                float weight = weights[k]; 
                // Activation of the input neuron.
                float input_activation = input[k].activation;
                // Answer to add to the summation.
                float answer = weight * input_activation + bias;
                // Add answer to the new activation.
                new_activation += answer;
            }

            // Set our new activation.
            activation = HyperbolicTangtent(new_activation);
        }

        /// <summary> Sigmoid function for activations </summary>
        public float LogSigmoid(float x)
        {
            if (x < -45.0) return 0.0f;
            else if (x > 45.0) return 1.0f;
            else return 1.0f / (1.0f + (float)Math.Exp(-x));
        }

        /// <summary> HyperbolicTangtent function for activations </summary>
        public float HyperbolicTangtent(float x)
        {
            if (x < -45.0f) return -1.0f;
            else if (x > 45.0f) return 1.0f;
            else return MathF.Tanh(x);
        }
    }

    public class NeuralNetwork
    {
        public Neuron[][] network;

        private Random random = new Random();

        // Contructors:
        public NeuralNetwork() {}
        public NeuralNetwork(int[] layers) => Setup(layers);

        /// <summary>
        /// Setup the network with n layers and n neurons.
        /// </summary>
        /// <param name="layers">The amount of layers and neurons.</param>
        /// <returns>Whether the setup was successful.</returns>
        public bool Setup(int[] layers)
        {
            // Check if network has atleast three layers.
            // (input, hidden, output)
            if (layers.Length < 3)
                return false;

            // Check if all layers have atleast one neuron.
            if (layers.Any(l => l <= 0))
                return false;

            // Create the network.
            Neuron[][] setup = new Neuron[layers.Length][];

            // Create the input layer.
            setup[0] = new Neuron[layers[0]];

            // Setup the input layer.
            for (int n = 0; n < setup[0].Length; n++)
                setup[0][n] = new Neuron();

            // Start at 1 because we don't need to setup the input layer.
            for (int l = 1; l < setup.Length; l++)
            {
                // Create the layer.
                setup[l] = new Neuron[layers[l]];

                for (int n = 0; n < setup[l].Length; n++)
                {
                    // Grab the neuron.
                    Neuron neuron = new Neuron
                    {
                        // bias = random number.
                        bias = (float)random.NextDouble() * 2 - 1,

                        // input = the previous layer.
                        input = setup[l - 1],

                        // weights.length = the size of the previous layer.
                        weights = new float[setup[l - 1].Length]
                    };

                    // weigths = random numbers.
                    for (int w = 0; w < neuron.weights.Length; w++)
                        neuron.weights[w] = (float)random.NextDouble() * 2 - 1;

                    // Put the neuron back where it came from.
                    setup[l][n] = neuron;
                }
            }

            // Assign the new network.
            network = setup;

            return true;
        }

        /// <summary>
        /// Randomize the neural network.
        /// </summary>
        public void Randomize()
        {
            // Start at 1 because we don't need to setup the input layer.
            for (int l = 1; l < network.Length; l++)
            {
                // Create the layer.
                network[l] = new Neuron[network[l].Length];

                for (int n = 0; n < network[l].Length; n++)
                {
                    // Grab the neuron.
                    Neuron neuron = new Neuron
                    {
                        // bias = random number.
                        bias = (float)random.NextDouble() * 10 - 5,

                        // input = the previous layer.
                        input = network[l - 1],

                        // weights.length = the size of the previous layer.
                        weights = new float[network[l - 1].Length]
                    };

                    // weigths = random numbers.
                    for (int w = 0; w < neuron.weights.Length; w++)
                        neuron.weights[w] = (float)random.NextDouble() * 10 - 5;

                    // Put the neuron back where it came from.
                    network[l][n] = neuron;
                }
            }
        }

        /// <summary>
        /// Sets the inputs of the network.
        /// </summary>
        /// <param name="inputs">The new inputs.</param>
        /// <returns>Whether the inputs were set.</returns>
        public bool SetInputs(float[] inputs)
        {
            // Check if network is not setup yet.
            if (network == null)
                return false;

            // Check if the input array is the correct size.
            if (inputs.Length != network[0].Length)
                return false;

            // Set the activations of the input layer.
            for (int i = 0; i < inputs.Length; i++)
                network[0][i].activation = inputs[i];
            return true;
        }

        /// <summary>
        /// Calculate the ouputs of the network.
        /// </summary>
        public void PropagateForwards()
        {
            // Check if network is not setup yet.
            if (network == null)
                return;

            // Start at 1 because we don't need to calculate the input layer.
            for (int l = 1; l < network.Length; l++)
            {
                // Loop over each neuron and calculate their activation.
                for (int n = 0; n < network[l].Length; n++)
                {
                    network[l][n].CalculateActivation();
                }
            }
        }

        /// <summary>
        /// Get the current outputs from the network.
        /// </summary>
        /// <returns>The current outputs.</returns>
        public float[] GetOutputs()
        {
            // Check if network is not setup yet.
            if (network == null)
                return null;

            return network[^1].Select(n => n.activation).ToArray();
        }

        /// <summary>
        /// Mutate the neural network.
        /// </summary>
        /// <param name="chance">Chance to mutate an attribute.</param>
        /// <param name="intensity">Intensity of the mutation.</param>
        public void Mutate(float chance, float intensity)
        {
            // Start at 1 because we don't need to calculate the input layer.
            for (int l = 1; l < network.Length; l++)
            {
                // Loop over each neuron and calculate their activation.
                for (int n = 0; n < network[l].Length; n++)
                {
                    // Shift the bias.
                    if ((float)random.NextDouble() < chance)
                        network[l][n].bias += ((float)random.NextDouble() * 2.0f - 1.0f) * intensity;

                    // Shift the weights.
                    for (int w = 0; w < network[l][n].weights.Length; w++)
                    {
                        if ((float)random.NextDouble() < chance)
                            network[l][n].weights[w] += ((float)random.NextDouble() * 2.0f - 1.0f) * intensity;
                    }
                }
            }
        }

        /// <summary>
        /// Creates a clone of this network.
        /// </summary>
        /// <returns>A clone of this network.</returns>
        public NeuralNetwork Clone()
        {
            NeuralNetwork clone = new NeuralNetwork();

            clone.network = (Neuron[][])network.Clone();

            return clone;
        }
    }
}
