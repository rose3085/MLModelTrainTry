using Microsoft.ML;
using PricePrediction.Model.PricePrediction;
using System.Data;

namespace PricePrediction
{
    internal static class program
    {

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);  // seed:0 more deterministic
            // seed is used to suffle the data before splitting

            string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
            string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
            string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "PricePredictionModel.zip");

            //load data
            var trainingData = mlContext.Data.LoadFromTextFile<TaxiTripPriceData>(_trainDataPath, hasHeader: true, separatorChar: ',');

            var hasModelSchemaChanged = HasModelSchemaChanged(mlContext,_modelPath,trainingData);
            //if (hasModelSchemaChanged)
            //{
                var model = Train(mlContext, trainingData, _modelPath);
                Evaluate(mlContext, _testDataPath, model);
                TestSinglePrediction(mlContext, model);
            //}
            //else 
            //{
            //    var model = mlContext.Model.Load(_modelPath, out _);
            //    TestSinglePrediction(mlContext, model);

            //}

           
        }

        static ITransformer Train(MLContext mlContext, IDataView trainingData, string modelPath)
        {
            //load data
            //var trainingData = mlContext.Data.LoadFromTextFile<TaxiTripPriceData>(dataPath, hasHeader:true, separatorChar:',');

            //create pipeline and train
            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                    // OneHotEncoding converts each text/number into a one hot encoded vector i.e one columns for each header 
                .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
               // now concat all columns in single features column.
               .Append(mlContext.Regression.Trainers.FastTree());
            // fastTree in Ml.Net is same as the RandomForest algorithm

            var model = pipeline.Fit(trainingData);

            mlContext.Model.Save(model, trainingData.Schema, modelPath);
            return model;

        }

        static void Evaluate(MLContext mlContext, string dataPath, ITransformer model)
        {
            var trainingData = mlContext.Data.LoadFromTextFile<TaxiTripPriceData>(dataPath, hasHeader: true, separatorChar: ',');
            var predictions = model.Transform(trainingData);
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");
            //evaluate the RootMeanSquare 
            // if RMS near 0 then model is more better
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:0.##}");

            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}"); // root square ,,, if near 1 then model is more better

        }

        static bool HasModelSchemaChanged(MLContext mlContext,string modelPath, IDataView newData)
        {
            if (!File.Exists(modelPath))
                return true;
            try
            {
                using var fileStream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read);
                var loadedModel = mlContext.Model.Load(fileStream, out var modelInputSchema);

                // Compare column count
                if (modelInputSchema.Count != newData.Schema.Count)
                    return true;

                // Compare each column name and type
                for (int i = 0; i < modelInputSchema.Count; i++)
                {
                    var modelCol = modelInputSchema[i];
                    var newCol = newData.Schema[i];

                    if (modelCol.Name != newCol.Name || modelCol.Type != newCol.Type)
                        return true;
                }

                return false;
            }
            catch
            {
                return true;
            }

        }
        static void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
            var taxiTripSample = new TaxiTripPriceData()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 600,
                TripDistance = 4.73f,
                PaymentType = "CRD",
               
            };

            var predictionFunction = mlContext.Model.CreatePredictionEngine<TaxiTripPriceData, TaxiTripPricePrediction>(model);
            var prediction = predictionFunction.Predict(taxiTripSample);
            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");
        }



    }
}