using System.Collections.Generic;

namespace NeuralNetworks
{
    public class Topology
    {
        public int InputCount { get; }
        public int OutputCount { get; }
        //коллекция количества нейронов на каждом скрытом слое. Скрытый это который ни входной, ни выходной 
        public List<int> HiddenLayers { get; }

        public Topology(int  inputCount, int outputCount, params int[] hiddenLayers)
        {
            InputCount = inputCount;
            OutputCount = outputCount;
            HiddenLayers = new List<int>(hiddenLayers);
        }
    }
}