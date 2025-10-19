using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks
{
    public class Layer
    {
        public List<Neural> Neurals { get; }
        public int Count => Neurals?.Count ?? 0;


        public Layer(List<Neural> neurals, NeuralType neuralType = NeuralType.Normal)
        {
            //todo проверить все входные данные на соответсвеи типу. Прогграммирование по контракту 
            if (!neurals.TrueForAll(n=>n.NeuralType == neuralType)) return;
            Neurals = neurals;
        }

        public List<double> GetSignals()
        {
            return Neurals.Select(neural => neural.Output).ToList();
        }
    }
}