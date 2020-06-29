"""
This file gathers the transforms used in "models" inside each pipeline.
"""
import os

os.system('pip install transformers')

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator
import re
from collections import Counter

import nltk
from nltk import ngrams
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from functools import reduce
from gensim.models import Word2Vec, KeyedVectors


from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer
from transformers import XLNetTokenizer
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AlbertConfig



class BaseTransform(ABC):
    """Abstract class for each transformer"""

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def transform(self, X):
        """Transforms original data and returns transformation"""
        pass

    def fit_transform(self, X, y):
        """Apply directly fit and transform methods on data X"""
        self.fit(X, y)
        return self.transform(X)


class FuncTransform(BaseTransform):
    """Transformer to apply any function to the data"""

    def __init__(self, f):
        self.f = f

    def fit(self, X, y):
        return self

    def transform(self, X):
        if type(X) == pd.Series:
            return X.apply(self.f)
        elif type(X) == pd.DataFrame:
            return X.applymap(self.f)


class POSTransform(BaseTransform):
    """Transformer to turn words into their POS tag"""

    def __init__(self, tags_only=True, join_labels=True):
        self.tags_only = tags_only
        self.join_labels = join_labels

    def fit(self, X, y):
        pass

    def transform(self, X):
        XPos = X.applymap(lambda x: nltk.word_tokenize(x))  # Tokenization of each sentence
        XPos = XPos.applymap(lambda x: nltk.pos_tag(x))  # Get POS tags for each sentence
        if self.tags_only:
            XPos = XPos.applymap(lambda x: list(zip(*x))[1])  # unzip the return to get only the tags
            if self.join_labels:
                XPos = XPos.applymap(lambda x: ' '.join(x))  # Transform list of tags into single string object
        return XPos


class SynonymTransform(BaseTransform):
    """Uses a synonyms dictionary to replace words with the same meaning by a unique synonym"""

    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform_element(self, L):
        tag_dict = {"J": wn.ADJ, "N": wn.NOUN, "V": wn.VERB, "R": wn.ADV}
        new_L = []
        for i, (word, tag) in enumerate(L):
            new_word = word
            tag_start = tag[0].upper()
            if tag_start in tag_dict:
                synsets = wn.synsets(word, tag_dict[tag_start])
                if synsets:
                    new_word = synsets[0].name().split('.')[0]
            new_L.append(new_word)
        return " ".join(new_L)

    def transform(self, X):
        X = POSTransform(tags_only=False).transform(X)
        return X.applymap(lambda elt: self.transform_element(elt))


class MinEditDistance(BaseTransform):
    """Computes minimum edit distance between two sentences"""

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def transform(self, X):
        Xmed = pd.Series([nltk.edit_distance(X.loc[i]['sentence1'],
                                             X.loc[i]['sentence2'],
                                             substitution_cost=2) for i in X.index],
                         index=X.index)
        return Xmed


class ProcessingVocab:
    """
    Build vocabulary over training set
    :param X: pd.DataFrame, data
    """

    def __init__(self, X, ngram_range=(1, 1), keep_stopwords=False, keep_specialchr=False, tokenize=True):
        """
        :param n: int, length of ngrams
        """
        self.X = pd.concat([X['sentence1'], X['sentence2']], axis=0)  # sentences

        self.stopwords = set(nltk.corpus.stopwords.words("english"))
        self.regex = re.compile('[a-zA-Z0-9]')

        self.keep_stopwords = keep_stopwords
        self.keep_specialchr = keep_specialchr
        self.ngram_range = ngram_range
        self.tokenize = tokenize

    def is_valid_token(self, word):
        """Check if word is valid regarding options"""
        word = word.lower()
        if not self.keep_stopwords:
            return word not in self.stopwords
        if not self.keep_specialchr:
            return self.regex.match(word)
        return True

    def clean_one(self, doc):
        """Pre-processing of one document"""
        if self.tokenize:
            tokens = [w.lower() for w in nltk.tokenize.word_tokenize(doc) if self.is_valid_token(w)]
        else:
            tokens = doc
        list_ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            if n == 1:
                list_ngrams += tokens
            else:
                list_ngrams += [' '.join(ngram) for ngram in list(ngrams(tokens, n))]
        return list_ngrams

    def clean_corpus(self):
        return [self.clean_one(doc) for doc in self.X]

    def full_voc(self):
        return Counter(ngram for doc in self.X for ngram in self.clean_one(doc))

    def voc(self):
        return {ngram: i for i, (ngram, count) in enumerate(self.full_voc().most_common())}


class LemmaTransform(BaseTransform):
    """Transforms each token into its lemma"""

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def fit(self, X, y):
        pass

    def transform(self, X):
        X = X.applymap(lambda x: ' '.join([self.wnl.lemmatize(t) for t in word_tokenize(x)]))
        return X


class TokenTransform(BaseTransform):
    """Tokenizes sentences"""

    def __init__(self, lowercase=True, keep_stopwords=False, keep_specialchr=False, lemmatize=True):
        self.lowercase = lowercase
        self.lemmatize = True
        self.keep_stopwords = keep_stopwords
        self.keep_specialchr = keep_specialchr
        self.wnl = WordNetLemmatizer()
        self.stopwords = set(nltk.corpus.stopwords.words("english"))
        self.regex = re.compile('[a-zA-Z0-9]')

    def fit(self, X, y):
        pass

    def transform(self, X):
        if self.lowercase:
            X = X.applymap(lambda x: x.lower())
        X = X.applymap(lambda x: word_tokenize(x))
        if not self.keep_stopwords:
            X = X.applymap(lambda x: [word for word in x if word not in self.stopwords])
        if self.lemmatize:
            X = X.applymap(lambda x: [self.wnl.lemmatize(word) for word in x])
        return X


class Vectorizer(BaseTransform):
    """Vectorizes each sentence with the specifieds weights"""

    def __init__(self, lowercase=True, stop_words=set(nltk.corpus.stopwords.words("english")),
                 ngram_range=(1, 1), ponderation='count'):
        super().__init__()
        self.lowercase = lowercase
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.ponderation = ponderation  # for transform

    def fit(self, X, y):
        self.vocab = ProcessingVocab(X).voc()  # Build vocabulary
        # Create vectorizer
        if self.ponderation == 'count':
            self.vectorizer = CountVectorizer(vocabulary=self.vocab, lowercase=self.lowercase,
                                              stop_words=self.stop_words, ngram_range=self.ngram_range)
        elif self.ponderation == 'tfidf':
            self.vectorizer = TfidfVectorizer(vocabulary=self.vocab, lowercase=self.lowercase,
                                              stop_words=self.stop_words, ngram_range=self.ngram_range)
        return self

    def transform(self, X):
        if not hasattr(self, 'vectorizer'):
            y = []
            self.fit(X, y)
        corpus = pd.concat([X['sentence1'], X['sentence2']])
        if self.ponderation == 'count':
            m = self.vectorizer.transform(corpus)
        elif self.ponderation == 'tfidf':
            m = self.vectorizer.fit_transform(corpus)  # fit needed to learn idf. But keep our vocab as columns. OK
        else:
            raise Exception("Please select appropriate ponderation in: 'tfidf', 'count'")

        dw1, dw2 = m[:m.shape[0] // 2], m[m.shape[0] // 2:]
        X1 = pd.DataFrame(dw1.toarray(), index=X.index, columns=self.vocab)
        X2 = pd.DataFrame(dw2.toarray(), index=X.index, columns=self.vocab)
        return X1, X2


class Mixor(BaseTransform):
    """
    Takes two embeddings 'sentence1', 'sentence2' and return one embedding for the paire
    """

    def __init__(self, method='sum'):
        """

        :param method: sum, product, substraction, concatenation: how to return one embedding for a couple of sentences
        """
        self.method = method

    def fit(self, X, y):
        return self

    def transform(self, X):
        if self.method == 'sum':
            R = reduce(lambda df1, df2: df1 + df2, X)
        elif self.method == 'sub':
            X1, X2 = X
            R = X1 - X2
        elif self.method == 'concatenate':
            R = pd.concat(X, axis=1)
        else:
            raise Exception("Please give possible method in sum, sub, concatenate")
        return R


class EmbeddingTransform(BaseTransform):
    """Transforms each token into a dense vector, according to the given embeddings (Glove)"""

    def __init__(self, fname="data/glove-wiki-gigaword-300.txt",
                 method=sum):
        self.fname = fname
        self.word_vectors = KeyedVectors.load_word2vec_format(self.fname, binary=False)
        self.method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X1 = pd.DataFrame(list(X['sentence1'].apply(lambda x: self.method([self.word_vectors[e]
                                                                           for e in
                                                                           filter(lambda w: w in self.word_vectors,
                                                                                  x)]))),
                          index=X.index)
        X2 = pd.DataFrame(list(X['sentence2'].apply(lambda x: self.method([self.word_vectors[e]
                                                                           for e in
                                                                           filter(lambda w: w in self.word_vectors,
                                                                                  x)]))),
                          index=X.index)
        return X1, X2


class EncoderTransform(BaseTransform):
    """Gives to each token an index, according to the vocabulary"""

    def __init__(self, vocab, padding=True, l_seq=None):
        vocab = {k: v + 1 for k, v in vocab.items()}
        self.word_index = vocab
        self.padding = padding
        self.l_seq = l_seq

    def fit(self, X, y=None):
        if not self.l_seq:
            self.l_seq = X.applymap(lambda x: len(x)).max().max()
        return self

    def transform(self, X):
        ind, col = X.index, X.columns
        X = X.applymap(lambda x: [self.word_index[word]
                                  for word in x
                                  if word in self.word_index])
        if self.padding:
            X = X.applymap(lambda x: pad_sequences([x], self.l_seq,
                                                   padding='post', truncating='post')[0])
        return X


class BERTProcessing(BaseEstimator):
    """Transformrs the data for the models based on BERT architectures"""

    def __init__(self, model_name, model_type):
        self.model_type = model_type
        self.model_name = model_name
        self.max_len = 128  # maximum length of a paire of sentence after tokenization.
        # If shorter, the paire of sentences will be padded, if longer, it will be truncated

        if self.model_type == 'xlnet':
            self.tokenizer = XLNetTokenizer.from_pretrained(self.model_name,
                                                            do_lower_case=False)  # we use a cased model for xlnet
        elif self.model_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name,
                                                           do_lower_case=True)  # we use an uncased model for bert
        elif model_type == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained(self.model_name,
                                                             do_lower_case=True)  # we use an uncased model for albert

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        sentences1 = X['sentence1'].values
        sentences2 = X['sentence2'].values

        # 1. Tokenization
        tokens_sent1 = [self.tokenizer.tokenize(sent) for sent in sentences1]
        tokens_sent2 = [self.tokenizer.tokenize(sent) for sent in sentences2]

        # 2. Replace tokens by their ids in the tokenizer dictionnary
        input_ids_sent1 = [self.tokenizer.convert_tokens_to_ids(x) for x in tokens_sent1]
        input_ids_sent2 = [self.tokenizer.convert_tokens_to_ids(x) for x in tokens_sent2]

        # 3. XLNet: Adding special tokens <cls> and <sep> ids to each paire of sentences: sen1 + <sep> + sen2 + <sep> + <cls>
        # 3. BERT: Adding special tokens [CLS] and [SEP] ids to each paire of sentences: [CLS] + sen1 + [SEP] + sen2 + [SEP]
        input_ids = [self.tokenizer.build_inputs_with_special_tokens(token_ids_0=paire[0],
                                                                     token_ids_1=paire[1])
                     for paire in zip(input_ids_sent1, input_ids_sent2)]

        # 4.1 Masking special tokens:
        # XLNet: <sep> and <cls>  (0 for sequence token, 1 for masked token)
        # BERT: [CLS] and [SEP] (0 for sentence token, 1 for masked token)
        # ALBERT: [CLS] and [SEP] (0 for a sequence token, 1 for a masked token)
        attention_masks = [self.tokenizer.get_special_tokens_mask(token_ids_0=paire_ids,

                                                                  already_has_special_tokens=True) for paire_ids in
                           input_ids]
        # 5. Padding:
        # XLNet : padding token is by default <pad>, index 5 in the dictionnary (tokenizer.pad_token_id)
        # BERT : padding token is by default [PAD], index 0 in the dictionnary (tokenizer.pad_token_id)
        pad_id = self.tokenizer.pad_token_id
        input_ids = pad_sequences(input_ids, maxlen=self.max_len, dtype="long", truncating="post", padding="post",
                                  value=pad_id)

        # 4.2 Masking padding token with the mask value 1 (padding mask vector with 1 until length of MAX_LEN):
        # XLNet: <pad>
        # BERT: [PAD]
        mask_value = 1
        attention_masks = pad_sequences(attention_masks,
                                        maxlen=self.max_len,
                                        dtype="long",
                                        truncating="post",
                                        padding="post",
                                        value=mask_value)

        # Inverse attention_masks values (1 => 0 and 0 => 1) for the model training (parameter attention_mask)
        # BERT: attention_mask must be: 1 for tokens that are NOT MASKED, 0 for MASKED tokens
        # ALBERT: attention_mask must be: 1 for tokens that are NOT MASKED, 0 for MASKED tokens
        # XLNet: attention_mask must be: 1 for tokens that are NOT MASKED, 0 for MASKED tokens.
        attention_masks = np.logical_not(attention_masks).astype('long')

        # 6.1 Create token_type_ids_ for paire of sequences:
        # XLNet : 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 2 | first sequence | second sequence | CLS segment ID
        # BERT: 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 | first sequence | second sequence
        token_type_ids = [self.tokenizer.create_token_type_ids_from_sequences(token_ids_0=paire[0],
                                                                              token_ids_1=paire[1]) for paire in
                          zip(input_ids_sent1, input_ids_sent2)]

        # 6.2 Padding step
        token_type_ids = pad_sequences(token_type_ids,
                                       maxlen=self.max_len,
                                       dtype="long",
                                       truncating="post",
                                       padding="post",
                                       value=pad_id)

        return input_ids, attention_masks, token_type_ids

