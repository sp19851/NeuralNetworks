using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
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
            /*var topology = new Topology(4, 1, 2);
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
            Debug.WriteLine(result.ToString());*/
            
            var dataSet = new List<Tuple<double, double[]>>
            {
                //Результат - Пациент болен - 1
                //            Пациент здоров - 0
                // Неправильная температура T
                // Хороший возраст A
                // Курит S
                // Правильно питается S
                //                                             T  A  S  F
                new Tuple<double, double[]> (0, new double[] { 0, 0, 0, 0 }),
                new Tuple<double, double[]> (0, new double[] { 0, 0, 0, 1 }),
                new Tuple<double, double[]> (1, new double[] { 0, 0, 1, 0 }),
                new Tuple<double, double[]> (0, new double[] { 0, 0, 1, 1 }),
                new Tuple<double, double[]> (0, new double[] { 0, 1, 0, 0 }),
                new Tuple<double, double[]> (0, new double[] { 0, 1, 0, 1 }),
                new Tuple<double, double[]> (1, new double[] { 0, 1, 1, 0 }),
                new Tuple<double, double[]> (0, new double[] { 0, 1, 1, 1 }),
                new Tuple<double, double[]> (1, new double[] { 1, 0, 0, 0 }),
                new Tuple<double, double[]> (1, new double[] { 1, 0, 0, 1 }),
                new Tuple<double, double[]> (1, new double[] { 1, 0, 1, 0 }),
                new Tuple<double, double[]> (1, new double[] { 1, 0, 1, 1 }),
                new Tuple<double, double[]> (1, new double[] { 1, 1, 0, 0 }),
                new Tuple<double, double[]> (0, new double[] { 1, 1, 0, 1 }),
                new Tuple<double, double[]> (1, new double[] { 1, 1, 1, 0 }),
                new Tuple<double, double[]> (1, new double[] { 1, 1, 1, 1 })
            };
            var topology = new Topology(4, 1, 0.1, 2);
            var neuralNetwork = new NeuralNetworks(topology);
            //обучаем
            var difference = neuralNetwork.Learn(dataSet, 100000);
            //используем
            /*foreach (var data in dataSet)
            {
                results.Add(neuralNetwork.FeedForward(data.Item2).Output);
            }*/
            var results = dataSet.Select(data => (neuralNetwork.FeedForward(data.Item2).Output)).ToList();

            for (var i = 0; i < results.Count; i++)
            {
                var expected = Math.Round(dataSet[i].Item1, 3);
                var actual = Math.Round(results[i], 3);
                //var actual = results[i];
                Assert.AreEqual(expected, actual);
            }
            
            
            
            
            //Debug.WriteLine(result.ToString());
        }
    }
}