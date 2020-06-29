"""
Estimators are implemented here (last step of the pipeline)

Exemple :
https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html

from transforms import MyTransform1, MyTransform2
from sklear.model_selection import Model
class MyModel(BaseEstimator):
    def __init__(self, params):
        Transformer1, Transformer2 = MyTransform1(), MyTransform2() # Définition des transformers (dérivent de BaseTransform)
        model = Model() # Définition du modèle final (dérive de BaseEstimator)
        self.pipeline = make_pipeline([Transformer1, Transformer2, model]) # Création de la pipeline
    def fit(self, X):
        pass
    def predict(self, X):
        return self.pipeline.predict(X)

"""

import os

os.system('pip install transformers')

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import nltk
from gensim.models import Word2Vec, KeyedVectors

import tensorflow as tf
from keras.layers import Dense, Embedding, Bidirectional, LSTM, \
    Input, dot
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, BertConfig
from transformers import XLNetForSequenceClassification, XLNetConfig
from transformers import AlbertForSequenceClassification, AlbertConfig
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from tqdm import trange

from scipy.stats import spearmanr

import transforms



def spearman_measure(preds, labels):
    """
    labels : list of labels
    preds : list of list, [[pred1], [pred2], ...]
    Used in BERT validation step
    """
    preds = np.array([i[0] for i in preds])
    labels = np.array(labels)
    return spearmanr(preds, labels)[0]


class CosineSimilarity(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        X1, X2 = X
        return pd.Series(np.diag(cosine_similarity(X1, X2)), X1.index)


class BowModel(BaseEstimator):
    """Bag of word model [BOW + cosine similarity]"""

    def __init__(self, lowercase=True, stop_words=set(nltk.corpus.stopwords.words("english")),
                 ngram_range=(1, 1), ponderation='tfidf'):
        self.lowercase = lowercase
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.ponderation = ponderation
        vectorizer = transforms.Vectorizer(lowercase=self.lowercase,
                                           stop_words=self.stop_words, ngram_range=self.ngram_range,
                                           ponderation=self.ponderation)
        model = CosineSimilarity()
        self.pipeline = make_pipeline(vectorizer, model)

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return self.pipeline.predict(X)


class MLPEstimator(BaseEstimator):
    def __init__(self):
        n_comp = 1000
        self.model = Sequential()
        self.model.add(Dense(50, input_dim=n_comp, activation='relu'))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(optimizer="adam", loss="mean_squared_error", metrics=['mse'])

    def fit(self, X, y):
        stop = EarlyStopping(monitor='val_accuracy', patience=2,
                             restore_best_weights=True)  # On réserve 15% des données pour la validation
        self.model.fit(X, y, epochs=20, validation_split=0.15, callbacks=[stop])
        return self

    def predict(self, X):
        return self.model.predict(X)


class LSTMestimator(BaseEstimator):

    def pearson_similarity():
        n_y_true = (y_true - K.mean(y_true[:])) / K.std(y_true[:])
        n_y_pred = (y_pred - K.mean(y_pred[:])) / K.std(y_pred[:])

        top = K.sum((n_y_true[:] - K.mean(n_y_true[:])) * (n_y_pred[:] - K.mean(n_y_pred[:])), axis=[-1, -2])
        bottom = K.sqrt(K.sum(K.pow((n_y_true[:] - K.mean(n_y_true[:])), 2), axis=[-1, -2]) * K.sum(
            K.pow(n_y_pred[:] - K.mean(n_y_pred[:]), 2), axis=[-1, -2]))

        result = top / bottom
        return K.mean(result)

    def __init__(self, embedding_matrix):
        self.embedding_matrix = embedding_matrix

        self.seq1 = Input(shape=(None,))
        self.seq2 = Input(shape=(None,))
        self.embedding_layer = Embedding(*self.embedding_matrix.shape, input_shape=(None,),
                                         weights=[embedding_matrix], trainable=True)
        self.lstm = LSTM(50, kernel_initializer='he_normal', kernel_regularizer='l2')
        self.final_layer = Dense(1, activation='sigmoid')

        self.embed1 = self.embedding_layer(self.seq1)
        self.embed2 = self.embedding_layer(self.seq2)
        self.output1 = self.lstm(self.embed1)
        self.output2 = self.lstm(self.embed2)

        self.main_output = dot([self.output1, self.output2], axes=1, normalize=True)
        self.model = Model(inputs=[self.seq1, self.seq2], outputs=[self.main_output])
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', pearson_similarity])

    def fit(self, X, y):
        self.l_seq = len(X['sentence1'].values[0])
        X1, X2 = np.zeros((len(X), self.l_seq)), np.zeros((len(X), self.l_seq))
        for i, x in enumerate(X['sentence1']):
            X1[i] = x
        for i, x in enumerate(X['sentence2']):
            X2[i] = x
        stop = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)
        self.model.fit([X1, X2], y, validation_split=0.15, epochs=25, batch_size=32, callbacks=[stop])
        return self

    def predict(self, X):
        X1, X2 = np.zeros((len(X), self.l_seq)), np.zeros((len(X), self.l_seq))
        for i, x in enumerate(X['sentence1']):
            X1[i] = x
        for i, x in enumerate(X['sentence2']):
            X2[i] = x
        return self.model.predict([X1, X2])


def build_matrix(vocab, fname):
    """
    Create embedding matrix for SiameseLSTM
    """
    vocab = {k: v + 1 for k, v in vocab.items()}
    word_vectors = KeyedVectors.load_word2vec_format(fname, binary=False)
    df = pd.DataFrame({i: word_vectors[word]
                       for word, i in vocab.items()
                       if word in word_vectors.vocab}, columns=list(range(max(vocab.values()) + 1)))
    return df.transpose().fillna(0)


class SiameseLSTM(BaseEstimator):
    def __init__(self, embedding_matrix):
        self.embedding_matrix = embedding_matrix
        self.optimizer = 'adam'
        self.dropout_rate = 0.15
        self.kernel_initializer = 'he_normal'
        self.lstm_units = 50
        self.activation = None
        self.l_seq = 10

        self.seq1 = Input(shape=(None,))
        self.seq2 = Input(shape=(None,))
        self.embedding_layer = Embedding(*self.embedding_matrix.shape, input_shape=(None,),
                                         weights=[self.embedding_matrix], trainable=False)
        self.embed1 = self.embedding_layer(self.seq1)
        self.embed2 = self.embedding_layer(self.seq2)
        self.lstm_layer1 = Bidirectional(
            LSTM(self.lstm_units, dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate,
                 kernel_initializer=self.kernel_initializer, return_sequences=True))
        self.lstm_layer2 = Bidirectional(
            LSTM(self.lstm_units, dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate,
                 kernel_initializer=self.kernel_initializer))

        self.output1 = self.lstm_layer1(self.embed1)
        self.output1 = self.lstm_layer2(self.output1)
        self.output2 = self.lstm_layer1(self.embed2)
        self.output2 = self.lstm_layer2(self.output2)
        self.main_output = dot([self.output1, self.output2], axes=1, normalize=True)
        self.model = Model(inputs=[self.seq1, self.seq2], outputs=[self.main_output])
        self.model.compile(optimizer=self.optimizer, loss='mse', metrics=['accuracy'])

    def fit(self, X, y):
        X1, X2 = np.zeros((len(X), self.l_seq)), np.zeros((len(X), self.l_seq))
        for i, x in enumerate(X['sentence1']):
            X1[i] = x
        for i, x in enumerate(X['sentence2']):
            X2[i] = x
        stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        self.model.fit([X1, X2], y, validation_split=0.1, epochs=25,
                       batch_size=32, callbacks=[stop], shuffle=True)
        return self

    def predict(self, X):
        X1, X2 = np.zeros((len(X), self.l_seq)), np.zeros((len(X), self.l_seq))
        for i, x in enumerate(X['sentence1']):
            X1[i] = x
        for i, x in enumerate(X['sentence2']):
            X2[i] = x
        return self.model.predict([X1, X2])


class BERTModel(BaseEstimator):
    """
    Fine tuning of BERT and XLNET for regression (adding one layer on top of the pre-trained model, then training for the STS task)

    ALBERT is implemented but the choice of hyper-parameters has to be improved

    Inspired from the tutorial : mccormickml.com/2019/07/22/BERT-fine-tuning/
    Documentation from: huggingface.co/transformers/
    """

    def __init__(self, model_name, model_type):
        """
        Hyper-parameters found with validation set:
        xlnet-large-casd : epoch = 4,  learning_rate = 1E-5, batch_size = 16, epsilon = 1e-6
        bert-large-uncased : epoch = 4,  learning_rate = 3E-5, batch_size = 16, epsilon = 1e-8
        ALBERT xxlarge-v2 large : epoch = 3,  learning_rate = 5E-5, batch_size = 8, epsilon = 1e-6 to be improved...
        """
        self.model_name = model_name
        self.model_type = model_type

        # Cf transformers library, batch of 16 or 32 is advised for training. For memory issues, we will take 16. Gradient accumulation step has not lead
        # to great improvment and therefore won't be used here.
        if model_type == 'albert':
            self.batch_size = 8
        else:
            self.batch_size = 16

        available_model_name = ["xlnet-large-cased", "bert-large-uncased", "albert-xlarge-v2"]
        available_model_type = ["bert", "xlnet", "albert"]

        if self.model_name not in available_model_name:
            raise Exception("Error : model_name should be in", available_model_name)
        if self.model_type not in available_model_type:
            raise Exception("Error : model_name should be in", available_model_type)

        # Load BertForSequenceClassification, the pretrained BERT model with a single linear regression layer on top of the pooled output
        # Load our fined tune model: ex: BertForSequenceClassification.from_pretrained('./my_saved_model_directory/')
        if self.model_type == 'bert':
            self.config = BertConfig.from_pretrained(self.model_name, num_labels=1)  # num_labels=1 for regression task
            self.model = BertForSequenceClassification.from_pretrained(self.model_name, config=self.config)
        elif self.model_type == 'xlnet':
            self.config = XLNetConfig.from_pretrained(self.model_name, num_labels=1)
            self.model = XLNetForSequenceClassification.from_pretrained(self.model_name, config=self.config)
        elif self.model_type == 'albert':
            self.config = AlbertConfig.from_pretrained(self.model_name, num_labels=1)
            self.model = AlbertForSequenceClassification.from_pretrained(self.model_name, config=self.config)
        self.model.cuda()

        if self.model_name == 'xlnet-large-cased':
            self.epochs = 4
            self.lr = 1e-5
            self.eps = 1e-6

        elif self.model_name == 'bert-large-uncased':
            self.epochs = 4
            self.lr = 3e-5
            self.eps = 1e-8

        elif self.model_name == 'albert-xxlarge-v2':
            self.epochs = 3
            self.lr = 5e-5
            self.eps = 1e-6

        self.max_grad_norm = 1.0  # Gradient threshold, gradients norms that exceed this threshold are scaled down to match the norm.

        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, eps=self.eps)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        torch.cuda.get_device_name(0)

    def fit(self, X, y=None):
        input_ids, attention_masks, token_type_ids = X
        labels = y.values

        # Use train_test_split to split our data into train and validation sets for training. 0.9/0.1 is chosen
        # Random seed is given to match indexes between train input_ids, token_types_ids and masks
        train_inputs, validation_inputs, train_labels, validation_labels = \
            train_test_split(input_ids, labels, random_state=2018, test_size=0.1)

        train_masks, validation_masks, _, _ = \
            train_test_split(attention_masks, input_ids, random_state=2018, test_size=0.1)

        train_token_type_ids, validation_token_type_ids, _, _ = \
            train_test_split(token_type_ids, input_ids, random_state=2018, test_size=0.1)

        # Convert all of our data into torch tensors, the required datatype for our model
        train_inputs = torch.tensor(train_inputs, dtype=torch.long)  # id : integers
        validation_inputs = torch.tensor(validation_inputs, dtype=torch.long)

        train_labels = torch.tensor(train_labels, dtype=torch.float)  # regression : labels are float
        validation_labels = torch.tensor(validation_labels, dtype=torch.float)

        train_masks = torch.tensor(train_masks, dtype=torch.long)
        validation_masks = torch.tensor(validation_masks, dtype=torch.long)

        train_token_type_ids = torch.tensor(train_token_type_ids, dtype=torch.long)
        validation_token_type_ids = torch.tensor(validation_token_type_ids, dtype=torch.long)

        # Create an iterator of our data with torch DataLoader for memory efficency.
        train_data = TensorDataset(train_inputs, train_masks, train_labels, train_token_type_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels,
                                        validation_token_type_ids)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=self.batch_size)

        num_training_steps = len(
            train_dataloader) * self.epochs  # nbr_epochs * len(train_dataloader) where len(train_dataloader) = len(train_data)// batch_size
        num_warmup_steps = 0.1 * num_training_steps
        warmup_proportion = float(num_warmup_steps) / float(num_training_steps)

        scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)  # PyTorch scheduler

        for _ in trange(self.epochs, desc="Epoch"):
            # 1. Training step
            # Set our model to training mode (as opposed to evaluation mode)
            self.model.train()

            # Tracking variables
            tr_loss, nb_tr_steps = 0, 0

            # Train on all our data for one epoch
            for step, batch in enumerate(train_dataloader):
                # Add batch to GPU
                batch = tuple(t.to(self.device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels, b_token_type_ids = batch

                # Forward pass
                # attention_mask : correct format because of process_data()
                outputs = self.model(b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask,
                                     labels=b_labels)
                loss = outputs[0]

                # Backward pass
                loss.backward()

                # AdamW optimizer: update parameters and take a step using the computed gradient
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()  # Perform a step of gradient descent
                scheduler.step()  # Update learning rate schedule
                self.model.zero_grad()  # Reset gradient

                # Update tracking variables
                tr_loss += loss.item()
                nb_tr_steps += 1
                print("Step {} out  of {}".format(nb_tr_steps, (num_training_steps // self.epochs)))
            print("Train loss: {}".format(tr_loss / nb_tr_steps))

            # 2. Validation step
            # Put model in evaluation mode to evaluate loss on the validation set
            self.model.eval()

            # Tracking variables
            nb_eval_steps, eval_accuracy = 0, 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:
                # Add batch to GPU
                batch = tuple(t.to(self.device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels, b_token_type_ids = batch
                # Telling the model not to compute or store gradients, saving memory and speeding up validation
                with torch.no_grad():
                    # Forward pass, calculate logit predictions
                    outputs = self.model(b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask)
                    logits = outputs[0]

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                tmp_eval_accuracy = spearman_measure(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1
            print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
        return self

    def predict(self, X):
        inputs_ids, attention_masks, token_type_ids = X

        prediction_inputs = torch.tensor(inputs_ids)
        prediction_masks = torch.tensor(attention_masks).to(torch.int64)
        prediction_token_type_ids = torch.tensor(token_type_ids).to(torch.int64)

        prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_token_type_ids)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=self.batch_size)

        self.model.eval()
        predictions = []

        for batch in prediction_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_token_type_ids = batch
            with torch.no_grad():
                outputs = self.model(b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask)
                logits = outputs[0]

            logits = logits.detach().cpu().numpy()
            predictions.append(logits)
        t = predictions.copy()  # each batch is an array of scores
        scores = np.concatenate(t).ravel().tolist()
        return scores
