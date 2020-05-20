import SemEval
from SemEval.dataset_readers.semeval_datareader import *
from SemEval.models.semeval_classifier_attention import *
from SemEval.predictors.semeval_predictor import *
import pandas as pd
import numpy as np
from datetime import datetime
predictor = SemEvalPredictor.from_path(
    "/Users/malolan/Documents/Research/Lopez/sentiment/twitter_sentiment_analysis/output copy/model.tar.gz",
    "semeval-predictor")


df = pd.read_csv("tweets.csv", engine="python")
# df.rename(columns={"full_text": "sentence"}, inplace=True)
print("Dataframe contains ", df.shape[0], " rows")
start = datetime.now()
df = predictor.predict_batch_df(df)
print("Time Taken ", datetime.now() - start)
df["sentiment"] = df["logits"].apply(lambda x: np.argmax(x["logits"], axis=-1))\
    .replace([0, 1, 2], ["neutral", "positive", "negative"])
df.to_csv("sentimentscovid.csv")
