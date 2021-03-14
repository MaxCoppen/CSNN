using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CSNN
{
    public class Trainer
    {
        public Random random = new Random();

        public async Task<NeuralNetwork> TrainNetwork(int generations, int population_size, int[] network_layers, Func<NeuralNetwork, float> method)
        {
            // Create the mother and father.
            Agent mother = new Agent(new NeuralNetwork(network_layers));
            Agent father = new Agent(new NeuralNetwork(network_layers));

            // Perform each generation.
            for (int g = 0; g < generations; g++)
            {
                // Create a new population for this generation.
                Agent[] population = CreatePopulation(mother, father, population_size);

                // Calculate the fitness for all the population.
                for (int p = 0; p < population_size; p++)
                    population[p].fitness = method.Invoke(population[p].network);

                // Get the parents for the next generation.
                Agent[] parents = population.OrderByDescending(p => p.fitness).Take(2).ToArray();
                mother = parents[0];
                father = parents[1];

                // Output debug to the console.
                Console.SetCursorPosition(0, 0);
                Console.WriteLine("generations: " + (g + 1));
                Console.WriteLine("loss: " + (1.0f - method.Invoke(mother.network)));

                if (g % 10 == 0)
                    await Task.Delay(1);
            }

            // Return the maximum fitness.
            return mother.network;
        }

        private Agent[] CreatePopulation(Agent mother, Agent father, int population_size)
        {
            Agent[] population = new Agent[population_size];

            // Put the parents into the next generation.
            population[0] = mother.Clone();
            population[1] = father.Clone();

            // Set the next 18 agents to be mutated versions of the parents.
            for (int p = 0; p < 9; p++)
            {
                population[2 + p * 2] = mother.MutatedClone(0.2f, 0.001f);
                population[3 + p * 2] = father.MutatedClone(0.2f, 0.001f);
            }

            // Set the other agents to be random.
            for (int p = 10; p < population_size; p++)
            {
                NeuralNetwork nn = mother.network.Clone();
                nn.Randomize();
                population[p] = new Agent(nn);
            }

            return population;
        }
    }

    public class Agent
    {
        public NeuralNetwork network;
        public float fitness;

        public Agent(NeuralNetwork nn) => network = nn;

        public Agent Clone() => new Agent(network.Clone());

        public Agent MutatedClone(float chance, float intensity)
        {
            Agent clone = new Agent(network.Clone());

            clone.network.Mutate(chance, intensity);

            return clone;
        }
    }
}
