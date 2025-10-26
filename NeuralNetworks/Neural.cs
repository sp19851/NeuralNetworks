using System;
using System.Collections.Generic;

namespace NeuralNetworks
{
    public class Neural
    {
        public List<double> Weights { get; }
        //Добавляем входные сигналы для обучения
        private List<double> Inputs { get; }
        public NeuralType NeuralType { get; }
        public double Output { get; private set; }
        //Добавляем дельту для обучения
        public double Delta { get; private set; }

        public Neural(int inputCount, NeuralType neuralType = NeuralType.Hidden)
        {
            NeuralType = neuralType;
            Weights = new List<double>();
            Inputs = new List<double>();
            //входной слой всегда имеет вес 1
            //инициализируем веса случайными числами
            InitWeightRandomValues(inputCount);
        }

        private void InitWeightRandomValues(int inputCount)
        {
            var rnd = new Random();
            for (var i = 0; i < inputCount; i++)
            {
                //у входных нейронов вес всегда 1
                if (NeuralType == NeuralType.Input)
                {
                    Weights.Add(1);
                }
                else
                {
                    Weights.Add(rnd.NextDouble());    
                }
                Inputs.Add(0);
            }
        }

        public double FeedForward(List<double> inputs)
        {
            var sum = 0.0;
            for (var i = 0; i < inputs.Count; i++)
            {
                sum += Weights[i] * inputs[i];
                //сохраняем входные сигналы
                Inputs[i] = inputs[i];
            }
            //Для входных нейронов сигмоидная функция не применяется
            Output = NeuralType != NeuralType.Input ? Sigmoid(sum) : sum;
            return Output;
        }

        private double Sigmoid(double x)
        {
            var result = 1.0 / (1.0 + Math.Pow(Math.E, -x));
            return result;
        }

        private double SigmoidDx(double x)
        {
            var sigmoid = Sigmoid(x);
            return sigmoid / (1-sigmoid);
        }

        /*public void SetWeights(params double[] weights)
        {
           //todo проверить соотвествие количекства весов и входов у нейрона 
           
           //todo Удалить помле добавления возможности обучения сети
           for (var i= 0; i < weights.Length; i++)
           {
               Weights[i] = weights[i];
           }
           
        }*/

        public void Lern(double error, double learningRate)
        {
            //входные нейроны не обучаются
            if (NeuralType == NeuralType.Input) return;
            Delta = error * SigmoidDx(Output);
            for (var i = 0; i < Weights.Count; i++)
            {
                var weight = Weights[i];
                var input = Inputs[i];
                var newWeight = weight - input * Delta * learningRate;
                Weights[i] = newWeight;
            }
        }
        
        
        public override string ToString()
        {
            return $"Output: {Output} Neuron Type: {NeuralType}";
        }
    }
}