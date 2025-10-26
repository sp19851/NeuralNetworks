using System.Collections.Generic;

namespace NeuralNetworks
{
    public class Topology
    {
        public int InputCount { get; }
        public int OutputCount { get; }

        public double LerningRate { get; }

        //Коллекция количества нейронов на каждом скрытом слое. Скрытый это который ни входной, ни выходной 
        public List<int> HiddenLayers { get; }

        public Topology(int  inputCount, int outputCount, double lerningRate, params int[] hiddenLayers)
        {
            InputCount = inputCount;
            OutputCount = outputCount;
            LerningRate = lerningRate;
            HiddenLayers = new List<int>(hiddenLayers);
        }
    }
}