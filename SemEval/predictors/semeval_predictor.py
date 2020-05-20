#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from overrides import overrides

from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors import Predictor
from overrides import overrides
import numpy as np
from typing import List


@Predictor.register('semeval-predictor')
class SemEvalPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(
            language='en_core_web_sm', pos_tags=True)

    def predict(self, sentence: str) -> str:
        print(sentence)
        logits = self.predict_json({"sentence": sentence})
        print(logits)
        label_id = np.argmax(logits["logits"], axis=-1)
        label = self._model.vocab.get_token_from_index(label_id, 'labels')
        return label

    def predict_batch_df(self, df):
        instances = [self._dataset_reader.text_to_instance(None, text=sentence) for sentence in df["sentence"]]
        df["logits"] = self.predict_batch_instance(instances)
        return df

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(None, text=sentence)
