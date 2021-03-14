using System;

namespace CSNN.Examples
{
    class Main
    {
        static readonly int[] layers = new int[] { 2, 2, 1 };

        public static NeuralNetwork Run()
        {
            // Setup the trainer.
            Trainer trainer = new Trainer();

            // Train the network.
            NeuralNetwork result = trainer.TrainNetwork(1000, 100, layers, Training).Result;

            // Output information.
            Console.WriteLine("fitness: " + Training(result));

            result.SetInputs(inputs[0]);
            result.PropagateForwards();
            Console.WriteLine("(0, 0) [" + result.GetOutputs()[0] + "] {-1}");
            result.SetInputs(inputs[1]);
            result.PropagateForwards();
            Console.WriteLine("(0, 1) [" + result.GetOutputs()[0] + "] { 1}");
            result.SetInputs(inputs[2]);
            result.PropagateForwards();
            Console.WriteLine("(1, 0) [" + result.GetOutputs()[0] + "] { 1}");
            result.SetInputs(inputs[3]);
            result.PropagateForwards();
            Console.WriteLine("(1, 1) [" + result.GetOutputs()[0] + "] {-1}");

            return result;
        }

        public static float[][] inputs = new float[][]
        {
            new float[] { 0, 0 },
            new float[] { 0, 1 },
            new float[] { 1, 0 },
            new float[] { 1, 1 }
        };

        public static float[] outputs = new float[]
        {
            -1,
            1,
            1,
            -1
        };

        public static float Training(NeuralNetwork nn)
        {
            float f = 0;

            for (int t = 0; t < 4; t++)
            {
                // Set the inputs and run the algorithm.
                nn.SetInputs(inputs[t]);
                nn.PropagateForwards();

                // Get the output of the network.
                float output = nn.GetOutputs()[0];

                // Add to the average fitness.
                f += (2.0f - MathF.Abs(output - outputs[t])) / 2.0f;
            }

            return f / 4.0f;
        }
    }
}
