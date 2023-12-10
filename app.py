from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from keyword_spacy import KeywordExtractor
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any
import numpy as np


class TextModel(BaseModel):
    text: str


def numpy_encoder(obj: Any):
    if isinstance(obj, np.float32):
        return float(obj)
    raise TypeError


app = FastAPI()

try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

nlp.add_pipe("keyword_extractor", last=True, config={"top_n": 100,
                                                     "top_n_sent": 100,
                                                     "min_ngram": 2,
                                                     "max_ngram": 3,
                                                     "strict": True})


@app.post("/extract-keywords/")
async def extract_keywords(text_model: TextModel):
    doc = nlp(text_model.text)

    keyword_data = [
        {
            "term": term,
            "frequency": frequency,
            "score": score
        }
        for term, frequency, score in doc._.keywords
    ]
    json_compatible_item_data = jsonable_encoder(
        keyword_data, custom_encoder={np.float32: numpy_encoder})

    return JSONResponse(content=json_compatible_item_data)
    # # Proses teks dengan Spacy untuk mendapatkan noun phrases
    # doc = nlp(text_model.text)
    # noun_phrases = [chunk.text.lower().strip()
    #                 for chunk in doc.noun_chunks if chunk.text.strip()]

    # # Menghitung TF-IDF skor untuk setiap noun phrase
    # vectorizer = TfidfVectorizer()
    # vectors = vectorizer.fit_transform(noun_phrases)
    # feature_names = vectorizer.get_feature_names_out()
    # scores = vectors.toarray().flatten()

    # # Membuat mapping antara noun phrases dan skor TF-IDF mereka
    # keywords_score = list(zip(noun_phrases, scores))

    # # Mengurutkan berdasarkan skor tertinggi
    # sorted_keywords = sorted(
    #     keywords_score, key=lambda tup: tup[1], reverse=True)

    # # Mengkonversi ke format yang diinginkan: array of dict dengan "term", "frequency", dan "score"
    # keyword_data = [
    #     {
    #         "term": keyword,
    #         "score": score,
    #     }
    #     for keyword, score in sorted_keywords
    # ]

    return keyword_data
