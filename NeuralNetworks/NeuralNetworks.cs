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

        public Neural FeedForward(List<double> inputSignals)
        {
            //todo проверить,что бы количество в оходных сигналов соотсветсвовало количеству нейронов на входном слое
            if (Layers.First().Count != inputSignals.Count) return null;
            SendSignalsToInputNeurals(inputSignals);
            
            FeedForwardAllLayersAfterInput();

            //если на выходном слое только один нейрон - возращаем его вес,
            //иначе сортируем выходные нейроны по убыванию весов и выозвращаем первый( с самым большим весом)
            return Topology.OutputCount == 1 ? Layers.Last().Neurals[0] : Layers.Last().Neurals.OrderByDescending(n=>n.Output).First();
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

        private void SendSignalsToInputNeurals(List<double> inputSignals)
        {
            for (var i = 0; i < inputSignals.Count; i++)
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
            for (var i = 0; i < Topology.InputCount; i++)
            {
                //выходные нейроны могут иметь разное количество выходов, в зависимости от количества нейронов предыдущем слое
                var neural = new Neural(lastLayer.Count, NeuralType.Output);
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
                    var neural = new Neural(lastLayer.Count);
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