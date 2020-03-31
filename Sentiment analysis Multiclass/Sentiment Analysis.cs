using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace Sentiment_analysis_Multiclass
{
    public class Sentiment_Analysis
    {
        [LoadColumn(0)]
        public int PhraseID { get; set; }
        [LoadColumn(1)]
        public int SentencesID { get; set; }
        [LoadColumn(2)]
        public string Phrase { get; set; }
        [LoadColumn(3)]
        public string Sentiment { get; set; }
        
       
    }
    public class Sentiments 
    {
        [ColumnName("PredictedLabel")]
        public string Sentiment;
    }
}
