using MLModelTrainTry.Model.SentimentAnalysis;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLModelTrainTry.TestData
{
   static class TestSentimentData
    {
        internal static readonly SentimentData sampleStatement = new SentimentData()
        {
            SentimentText = "This was a very bad steak",
        };
    }
}
