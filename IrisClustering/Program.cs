using IrisClustering.Model.IrisClustering;
using IrisClustering.TestData;
using Microsoft.ML;
using System.Reflection;

namespace IrisClustering
{
    internal static class program
    {

        static void Main(string[] args)
        {
            string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.data");
            string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IrisClusteringModel.zip");
            MLContext mlContext = new MLContext();
            var dataView = LoadData(mlContext, _dataPath);
            var hasModelSchemaChanged = HasModelSchemaChanged(mlContext, _modelPath, dataView);
            if (hasModelSchemaChanged)
            {
                var model = BuildAndTrainModel(mlContext,_modelPath,dataView);
                Evaluate(mlContext,dataView ,model);
                CreatePrediction(mlContext,model);
            }
            else 
            {
                var model = mlContext.Model.Load(_modelPath, out _);
                CreatePrediction(mlContext, model);
            }

        }

        static IDataView LoadData(MLContext mLContext, string dataPath)
        {
            var dataView = mLContext.Data.LoadFromTextFile<IrisData>(dataPath, hasHeader: false, separatorChar: ',');
            return dataView;
        }

        static ITransformer BuildAndTrainModel(MLContext mLContext,string modelPath, IDataView dataView)
        {
            var pipeline = mLContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                                                .Append(mLContext.Clustering.Trainers.KMeans("Features",numberOfClusters:3));
            var model = pipeline.Fit(dataView);
            mLContext.Model.Save(model, dataView.Schema, modelPath);
            return model;
        }

        static void Evaluate(MLContext mLContext, IDataView dataView,ITransformer model)
        {
            var predictions = model.Transform(dataView);
            var metrics = mLContext.Clustering.Evaluate(predictions);
        }

        static void CreatePrediction(MLContext mlContext, ITransformer model)
        {
            var predictor = mlContext.Model.CreatePredictionEngine<IrisData, IrisPrediction>(model);
            var prediction = predictor.Predict(TestIrisData.Setosa);
            Console.WriteLine($"Cluster: {prediction.PredictedClusterId}");
            Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances ?? Array.Empty<float>())}");
        }

        static bool HasModelSchemaChanged(MLContext mlContext, string modelPath, IDataView newData)
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

    }
}
