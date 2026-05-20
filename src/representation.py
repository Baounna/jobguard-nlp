"""
Text representation module.

Word2Vec is provided by gensim when available.
If gensim is not installed (e.g. Python 3.14 wheels missing),
we fall back to LSA (TruncatedSVD on TF-IDF), which produces
equivalent dense semantic document vectors.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import joblib

try:
    from gensim.models import Word2Vec as _GensimW2V
    _GENSIM_AVAILABLE = True
except ImportError:
    _GENSIM_AVAILABLE = False

# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

def build_tfidf(texts, max_features=15000, ngram_range=(1, 2)):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2
    )
    X = vectorizer.fit_transform(texts)
    return vectorizer, X

def transform_tfidf(vectorizer, texts):
    return vectorizer.transform(texts)

# ---------------------------------------------------------------------------
# Word2Vec (gensim) — only available when gensim can be installed
# ---------------------------------------------------------------------------

def build_word2vec(texts, vector_size=100, window=5, min_count=2, workers=4):
    """Train a Word2Vec model (requires gensim)."""
    if not _GENSIM_AVAILABLE:
        raise ImportError(
            "gensim is not installed. Use build_lsa() for dense document vectors."
        )
    tokenized = [text.split() for text in texts]
    model = _GensimW2V(
        tokenized,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=15,
        seed=42
    )
    return model

def text_to_w2v_vector(w2v_model, text, vector_size=100):
    tokens = text.split()
    vectors = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(vector_size)

def texts_to_w2v_matrix(w2v_model, texts, vector_size=100):
    return np.array([text_to_w2v_vector(w2v_model, t, vector_size) for t in texts])

# ---------------------------------------------------------------------------
# LSA — pure scikit-learn dense semantic vectors (Word2Vec alternative)
# ---------------------------------------------------------------------------

def build_lsa(texts, n_components=100, max_features=15000):
    """
    Build LSA (Latent Semantic Analysis) document vectors.
    TF-IDF → TruncatedSVD yields dense semantic embeddings
    similar in spirit to Word2Vec averaged document vectors.
    Works with any Python / scikit-learn version.
    n_components is automatically capped to n_features - 1 so it
    works on both small synthetic data and the full Kaggle dataset.
    """
    tfidf = TfidfVectorizer(max_features=max_features, sublinear_tf=True, min_df=2)
    X_tfidf = tfidf.fit_transform(texts)
    n_components = min(n_components, X_tfidf.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    pipe = Pipeline([('tfidf', tfidf), ('svd', svd)])
    # svd was already fitted on X_tfidf; rebuild as proper fitted pipeline
    from sklearn.utils.validation import check_is_fitted
    X_lsa = svd.fit_transform(X_tfidf)
    # Store fitted steps so transform() works later
    pipe.steps[0] = ('tfidf', tfidf)
    pipe.steps[1] = ('svd', svd)
    return pipe, X_lsa

def transform_lsa(pipe, texts):
    return pipe.transform(texts)

def gensim_available():
    return _GENSIM_AVAILABLE

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_vectorizer(obj, path):
    joblib.dump(obj, path)

def load_vectorizer(path):
    return joblib.load(path)
