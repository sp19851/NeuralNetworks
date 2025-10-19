using System;
using System.Collections.Generic;

namespace NeuralNetworks
{
    public class Neural
    {
        public List<double> Weights { get; }
        public NeuralType NeuralType { get; }
        public double Output { get; private set; }

        public Neural(int inputCount, NeuralType neuralType = NeuralType.Normal)
        {
            NeuralType = neuralType;
            Weights = new List<double>();

            for (var i = 0; i < inputCount; i++)
            {
                Weights.Add(1);
            }
        }

        public double FeedForward(List<double> inputs)
        {
            var sum = 0.0;
            for (var i = 0; i < inputs.Count; i++)
            {
                sum += Weights[i] * inputs[i];
            }

            Output = Sigmoid(sum);
            return Output;
        }

        private double Sigmoid(double x)
        {
            var result = 1.0 / (1.0 + Math.Pow(Math.E, -x));
            return result;
        }

        public void SetWeights(params double[] weights)
        {
           //todo проверить соотвествие количекства весов и входов у нейрона 
           
           //todo Удалить помле добавления возможности обучения сети
           /*for (var i= 0; i < weights.Length; i++)
           {
               Weights[i] = (weights[i]);
           }*/
           Weights.AddRange(weights);
        }
        
        public override string ToString()
        {
            return $"Output: {Output} Neuron Type: {NeuralType}";
        }
    }
}