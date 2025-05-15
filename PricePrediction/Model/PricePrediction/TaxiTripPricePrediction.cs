using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PricePrediction.Model.PricePrediction
{
    public class TaxiTripPricePrediction : TaxiTripPriceData
    {

        [ColumnName("Score")]
        public float FareAmount;
    }
}
