using Microsoft.ML;
using Microsoft.ML.Data;
using MLModelTrainTry.Model.SentimentAnalysis;
using System.Reflection;
using static Microsoft.ML.DataOperationsCatalog;

namespace MLModelTrainTry
{

    internal static class program {

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            
            //load training data
            var splitDataView = LoadData(mlContext).TestSet;  //.TestSet conver splitview to IDataview
            IDataView trainingData = splitDataView;

            //check if we need to rebuilt the model
            string dataModelPath = Path.Combine(Environment.CurrentDirectory, "MlModels", "SentimentModel.zip");
            var hasSchemaChanged = HasSchemaChanged(mlContext, dataModelPath, trainingData);

            //create training pipeline and train model
            if (hasSchemaChanged)
            {
                var model = BuildAndTrainModel(mlContext, splitDataView);

                //evaluate the model
                Evaluate(mlContext, model, splitDataView);

                //save the models
                var modelPath = GetModelPath();
                mlContext.Model.Save(model, trainingData.Schema, modelPath);
                //create new prediction
                CreatePrediction(mlContext, model);
            }
            else
            {
                //use existing model
                var modelPath = GetModelPath();
                var model = mlContext.Model.Load(modelPath, out _);

                //create new prediction
                CreatePrediction(mlContext, model);
            }

        }

        static TrainTestData LoadData(MLContext mlContext)
        {

            string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            return splitDataView;
        }

        static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
            // create the pipeline for data processing
            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                                                     .Append(trainer);
            var model = estimator.Fit(splitTrainSet);
            return model;
        }

        static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            IDataView predictions = model.Transform(splitTestSet);
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            //ConsoleHelper.PrintBinaryClassificationMetrics(trainer.ToString(), metrics);

        }

        static void CreatePrediction(MLContext mlContext, ITransformer model)
        {
            var sampleStatement = new SentimentData()
            {
                SentimentText = "This was a very bad steak",
            };
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
            var resultPrediction = predictionFunction.Predict(sampleStatement);
            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }


        static bool HasSchemaChanged(MLContext mlContext,string modelPath, IDataView newData)
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
        static string GetModelPath()
        {
            string modelsFolder = Path.Combine(Directory.GetCurrentDirectory(), "MlModels");
            if (!Directory.Exists(modelsFolder))
            {
                Directory.CreateDirectory(modelsFolder);
            }
            string modelFileName = "SentimentModel.zip"; 
            string ModelPath = Path.Combine(modelsFolder, modelFileName);
            return ModelPath;
        }
       
    }


}




