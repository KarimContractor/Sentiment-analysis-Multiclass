using System;
using System.IO;
using System.Linq;
using Microsoft.ML;

namespace Sentiment_analysis_Multiclass
{
    class Program
    {
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _trainDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "train.tsv");
        private static string _testDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "SentimentTest.csv");
        private static string _modelPath => Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

        private static MLContext _mLContext;

        private static PredictionEngine<Sentiment_Analysis, Sentiments> _predEngine;
        private static ITransformer _trainedmodel;
        static IDataView _trainingDataView;
        static IDataView train;
        static IDataView test;
        
        static void Main(string[] args)
        {
            _mLContext = new MLContext(seed:0);
            _trainingDataView = _mLContext.Data.LoadFromTextFile<Sentiment_Analysis>(_trainDataPath, hasHeader: true);
            var Split = _mLContext.Data.TrainTestSplit(_trainingDataView, testFraction: 0.015, samplingKeyColumnName: default);
            train = Split.TrainSet;
            test = Split.TestSet;


            var pipeline = ProcessData();
            //var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);
            var trainingPipeline = BuildAndTrainModel(test, pipeline);
            //Evaluate(_trainingDataView.Schema);
            Evaluate(test.Schema);

        }
        public static IEstimator<ITransformer> ProcessData() 
        {
            var pipeline = _mLContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Sentiment", outputColumnName: "Label")
                .Append(_mLContext.Transforms.Text.FeaturizeText(inputColumnName: "Phrase", outputColumnName: "textfeaturized"))
                .Append(_mLContext.Transforms.Concatenate("Features", "textfeaturized"))
                .AppendCacheCheckpoint(_mLContext);
            return pipeline;

        }
        public static IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline) 
        {
            var trainingPipline = pipeline.Append(_mLContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(_mLContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            _trainedmodel = trainingPipline.Fit(trainingDataView);

            _predEngine = _mLContext.Model.CreatePredictionEngine<Sentiment_Analysis, Sentiments>(_trainedmodel);
            Sentiment_Analysis se = new Sentiment_Analysis()
            {

                Phrase = //"This terms candidates are very normal"
                "This website sucks"
            };
            var prediction = _predEngine.Predict(se);
            Console.WriteLine($"single line prediction { prediction.Sentiment}");
            return trainingPipline;
        }
        public static void Evaluate(DataViewSchema trainingDataViewSchema)
        {
            
            //var testDataView = _mLContext.Data.LoadFromTextFile<Sentiment_Analysis>(_testDataPath,separatorChar:',', hasHeader: true);
            var testMetrics = _mLContext.MulticlassClassification.Evaluate(_trainedmodel.Transform(test));
            Console.WriteLine($"Metrics For Multi-Class Classification Model For Test Data");
            Console.WriteLine($"MicroAccuracy : {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"MacroAccuracy : {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"LogLoss : {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"LogLossReduction : {testMetrics.LogLossReduction:#.###}");
            SaveModelAsFile(_mLContext, trainingDataViewSchema, _trainedmodel);

        }
        private static void SaveModelAsFile(MLContext mLContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            _mLContext.Model.Save(model, trainingDataViewSchema, _modelPath);
        }
        private static void PredictIssue() 
        {
            ITransformer loadedModel = _mLContext.Model.Load(_modelPath, out var modelInputSchema);
            Sentiment_Analysis singlesentiment = new Sentiment_Analysis() { Phrase = "" };
            _predEngine = _mLContext.Model.CreatePredictionEngine<Sentiment_Analysis, Sentiments>(loadedModel);
            var prediction = _predEngine.Predict(singlesentiment);
            Console.WriteLine($"{prediction.Sentiment}");
        }

    }
}
