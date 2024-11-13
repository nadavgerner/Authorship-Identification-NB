import os
from nb.utils.load_data import build_dataframe_from_docs_txt
from nb.nb import NaiveBayes
import numpy as np
import pytest

# --- Test 1: Priors

def test_nb_priors():
    test_filepath = 'tests/data/docs.txt'
    a = 1

    df_train = build_dataframe_from_docs_txt(test_filepath)

    expected_priors = {
        '+': 2/5,
        '-': 3/5 
    }

    nb = NaiveBayes(alpha=a)
    nb.train_nb(df_train)

    # Test Priors
    assert nb.priors == pytest.approx(expected_priors), "Priors calculation is incorrect"


# --- Test 2: Liklihoods

def test_nb_likelihoods():

    test_filepath = 'tests/data/docs.txt'
    a = 1

    df_train = build_dataframe_from_docs_txt(test_filepath)

    nb = NaiveBayes(alpha = a)
    nb.train_nb(df_train)

    # The following comes from a manual calculation and counting:

    vocab_size = 20

    neg_word_counts = {
        'just': 1, 'plain': 1, 'boring': 1,
        'entirely': 1, 'predictable': 1, 'and': 2,
        'lacks': 1, 'energy': 1, 'no': 1,
        'surprises': 1, 'few': 1, 'laughs': 1,
        'very': 1
    }
    neg_total_words = sum(neg_word_counts.values()) # = 14

    pos_word_counts = {
        'very': 1, 'powerful': 1,
        'the': 2, 'most': 1, 'fun': 1,
        'film': 1, 'of': 1, 'summer': 1,
    }
    pos_total_words = sum(pos_word_counts.values())  # = 9

    neg_denom = neg_total_words + a * vocab_size  # = 34
    pos_denom = pos_total_words + a * vocab_size  # = 29

    vocab = ['summer', 'very', 'boring', 'fun', 'surprises', 
             'and', 'just', 'laughs', 'the', 'most', 'entirely', 
             'lacks', 'film', 'energy', 'predictable', 'few', 'of', 
             'powerful', 'plain', 'no']
    

    neg_counts = np.zeros(vocab_size)
    pos_counts = np.zeros(vocab_size)
                          
    for i, word in enumerate(vocab):
        neg_counts[i] = neg_word_counts.get(word, 0)
        pos_counts[i] = pos_word_counts.get(word, 0)

    expected_likelihoods = np.zeros((2, vocab_size))

    expected_likelihoods[0, :] = (neg_counts + a) / neg_denom

    expected_likelihoods[1, :] = (pos_counts + a) / pos_denom

    assert np.allclose(np.sort(nb.likelihoods), np.sort(expected_likelihoods), atol=1e-6), "Likelihood arrays do not match expected values"



# --- Test 3: Likelihoods sum to 1

def test_nb_likelihoods_sum():
    test_filepath = 'tests/data/docs.txt'
    a = 1

    df_train = build_dataframe_from_docs_txt(test_filepath)

    nb = NaiveBayes(alpha=a)
    nb.train_nb(df_train)

    # Test Likelihoods Sum to One
    assert nb.likelihoods.sum(axis=1) == pytest.approx(1.0, abs=1e-6), "Likelihoods do not sum to 1"


    