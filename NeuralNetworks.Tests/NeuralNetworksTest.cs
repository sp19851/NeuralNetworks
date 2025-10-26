using System.Collections.Generic;
using System.Diagnostics;
using JetBrains.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworks;

namespace NeuralNetworks.Tests
{
    [TestClass]
    [TestSubject(typeof(NeuralNetworks))]
    public class NeuralNetworksTest
    {

        [TestMethod]
        public void FeedForwardTest()
        {
            var topology = new Topology(4, 1, 2);
            var neuralNetwork = new NeuralNetworks(topology);
            //Задаем начальные веса скрытым нейронам в скрытых слоях. Скрытый слой в этой топологии один
            //сами веса почти от балды
            neuralNetwork.Layers[1].Neurals[0].SetWeights(0.5, -0.1, 0.3, -0.1);
            neuralNetwork.Layers[1].Neurals[1].SetWeights(0.1, -0.3, 0.7, -0.3);
            neuralNetwork.Layers[2].Neurals[0].SetWeights(1.2, 0.8);
            //Подаем сигналы на входные нейроны.
            //Первый (температура). 1 - повышенная.
            //Второй (возраст). 0 - ненормальный, до 16 или старше 25 лет.
            //Третий (курение). 0 - не курит.
            //Четвертый (питание). 0 - ненормальное.
            var result = neuralNetwork.FeedForward(new List<double> {1, 0, 0, 0 });
            Debug.WriteLine(result.ToString());
        }
    }
}