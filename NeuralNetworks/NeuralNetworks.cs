using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks
{
    public class NeuralNetworks
    {
        public Topology Topology { get; }
        public List<Layer> Layers { get; }

        public NeuralNetworks(Topology topology)
        {
            Topology = topology;
            Layers = new List<Layer>();

            CreateInputLayer();
            CreateHiddenLayers();
            CreateOutputLayer();
            
        }

        public Neural FeedForward(params double[] inputSignals)
        {
            //todo проверить,что бы количество в оходных сигналов соотсветсвовало количеству нейронов на входном слое
            if (Layers.First().NeuralCount != inputSignals.Length) return null;
            SendSignalsToInputNeurals(inputSignals);
            
            FeedForwardAllLayersAfterInput();

            //если на выходном слое только один нейрон - возращаем его вес,
            //иначе сортируем выходные нейроны по убыванию весов и выозвращаем первый( с самым большим весом)
            return Topology.OutputCount == 1 ? Layers.Last().Neurals[0] : Layers.Last().Neurals.OrderByDescending(n=>n.Output).First();
        }

        public double Learn(List<Tuple<double, double[]>> dataSet, int epoch)
        {
            var error = 0.0;
            for (var i = 0; i < epoch; i++)
                foreach (var data in dataSet)
                    error += BackPropagation(data.Item1, data.Item2);
            //возвращаем среднюю ошибку
            return error/epoch;
        }
        private double BackPropagation(double expectedOutput, params double[] inputSignals)
        {
            var actual = FeedForward(inputSignals).Output;
            var difference = actual - expectedOutput;
            foreach (var neural in Layers.Last().Neurals)
            {
                neural.Lern(difference, Topology.LerningRate);
            }

            //Так как один слой уже обучен, берем - 2
            for (var i = Layers.Count - 2; i >= 0; i--)
            {
                var layer = Layers[i];
                var previousLayer = Layers[i + 1];
                //перебираем нейроны слоя
                for (var j = 0; j < layer.NeuralCount; j++)
                {
                    var neural = layer.Neurals[j];
                    //вложенный цикл для каждого входа - если нейронов на предыдущем слое больше одного
                    for (var k = 0; k < previousLayer.NeuralCount; k++)
                    {
                        var previousNeural = previousLayer.Neurals[k];
                        var error = previousNeural.Weights[j] * previousNeural.Delta;
                        neural.Lern(error, Topology.LerningRate);
                    }
                }
            }
            //возвращается всегда квадратичная ошибка
            return Math.Pow(difference, 2);
        }
        
        
        private void FeedForwardAllLayersAfterInput()
        {
            for (var i = 1; i < Layers.Count; i++)
            {
                var previousLayerSignals = Layers[i-1].GetSignals();
                var layer = Layers[i];
                foreach (var neural in layer.Neurals)
                {
                    neural.FeedForward(previousLayerSignals);
                }
            }
        }

        private void SendSignalsToInputNeurals(params double[] inputSignals)
        {
            for (var i = 0; i < inputSignals.Length; i++)
            {
                var signal = new List<double>{inputSignals[i]};
                var neural = Layers.First().Neurals[i];
                neural.FeedForward(signal);
            }
        }

        private void CreateOutputLayer()
        {
            var outputNeurals = new List<Neural>();
            var lastLayer = Layers.Last();
            for (var i = 0; i < Topology.OutputCount; i++)
            {
                //выходные нейроны могут иметь разное количество выходов, в зависимости от количества нейронов предыдущем слое
                var neural = new Neural(lastLayer.NeuralCount, NeuralType.Output);
                outputNeurals.Add(neural);
            }
            var outputLayer = new Layer(outputNeurals, NeuralType.Output);
            Layers.Add(outputLayer);
        }

        private void CreateHiddenLayers()
        {
            for (var j = 0; j < Topology.HiddenLayers.Count; j++)
            {
                var hiddenNeurals = new List<Neural>();
                var lastLayer = Layers.Last();
                for (var i = 0; i < Topology.HiddenLayers[j]; i++)
                {
                    //выходные нейроны могут иметь разное количество выходов, в зависимости от количества нейронов предыдущем слое
                    var neural = new Neural(lastLayer.NeuralCount);
                    hiddenNeurals.Add(neural);
                }
                var hiddenLayer = new Layer(hiddenNeurals);
                Layers.Add(hiddenLayer);
            }
          
        }

        private void CreateInputLayer()
        {
            var inputNeurals = new List<Neural>();
            for (var i = 0; i < Topology.InputCount; i++)
            {
                //входные нейроны всегд аимею только один вход
                var neural = new Neural(1, NeuralType.Input);
                inputNeurals.Add(neural);
            }
            var inputLayer = new Layer(inputNeurals, NeuralType.Input);
            Layers.Add(inputLayer);
        }
    }
}