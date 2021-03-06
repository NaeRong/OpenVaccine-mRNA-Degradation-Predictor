# -*- coding: utf-8 -*-
"""OpenVaccine: COVID-19 mRNA

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12DdVwPfy9hZMLb6qd8FdN_07f31o0iQZ
"""

import zipfile
from google.colab import drive

drive.mount('/content/drive/')

import warnings
warnings.filterwarnings('ignore')

#the basics
import pandas as pd, numpy as np
import math, json, gc, random, os, sys
from matplotlib import pyplot as plt
import seaborn as sns


#tensorflow deep learning basics
import tensorflow as tf
#LSTM
#import tensorflow_addons as tfa

import tensorflow.keras.layers as L

from sklearn.model_selection import train_test_split
#LSTM from sklearn.model_selection importKFold

#Process data from local
import pandas as pd
import zipfile

zf = zipfile.ZipFile('/content/drive/My Drive/Colab Notebooks/sample_submission.csv.zip') 
zf_1 = zipfile.ZipFile('/content/drive/My Drive/Colab Notebooks/test.json.zip') 
zf_2 = zipfile.ZipFile('/content/drive/My Drive/Colab Notebooks/train.json.zip') 
sample_sub = pd.read_csv(zf.open('sample_submission.csv'))
test = pd.read_json(zf_1.open('test.json'), lines=True)
train = pd.read_json(zf_2.open('train.json'), lines=True)
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}

print(train.shape)
if ~ train.isnull().values.any(): print('No missing values')
train.head()

print(test.shape)
if ~ test.isnull().values.any(): print('No missing values')
test.head()

print(sample_sub.shape)
if ~ sample_sub.isnull().values.any(): print('No missing values')
sample_sub.head()

print(f"Train Columns: {train.columns.values}")
print(f"Test Columns: {test.columns.values}")
print(f"Submission Columns: {sample_sub.columns.values}")

"""## Understand the mRNA sequence"""

seq_train_lengths = []
for i in range(train.shape[0]):
    seq_train_lengths.append(len(train.sequence.values[i]))


seq_test_lengths = []
for i in range(test.shape[0]):
    seq_test_lengths.append(len(test.sequence.values[i]))

print(set(seq_train_lengths),set(seq_test_lengths))

from collections import Counter
print(train.seq_length.value_counts(),'\nFrequency:',Counter(train.head(1).sequence.values[0]))

test.seq_length.value_counts()

sns.countplot(test['seq_length'].value_counts())
ax.set_title('Sequence Length in test set')

"""## Structure column"""

train.structure.head(1).values[0]

Counter(train.structure.head(1).values[0])

train.structure.value_counts()

test.structure.value_counts()

struc_train_lengths = []
for i in range(train.shape[0]):
    struc_train_lengths.append(len(train.structure.values[i]))


struc_test_lengths = []
for i in range(test.shape[0]):
    struc_test_lengths.append(len(test.structure.values[i]))

set(struc_train_lengths)

set(struc_test_lengths)

"""## Signal to Noise feature"""

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10, 3))
ax = sns.distplot(train['signal_to_noise'])
ax.set_title('Signal to Noise feature (train)')
plt.show()

sns.countplot(train['SN_filter'])
ax.set_title('Signal / Noise filter distribution (train)')

print(f"Samples with signal_to_noise greater than 1: {len(train.loc[(train['signal_to_noise'] > 1 )])}")
print(f"Samples with SN_filter = 1: {len(train.loc[(train['SN_filter'] == 1 )])}")

"""## GRU Models

Tensorflow : tf.keras.layers.GRU formats

tf.keras.layers.GRU(
    units, activation='tanh', recurrent_activation='sigmoid', use_bias=True,
    kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
    bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    recurrent_constraint=None, bias_constraint=None, dropout=0.0,
    recurrent_dropout=0.0, implementation=2, return_sequences=False,
    return_state=False, go_backwards=False, stateful=False, unroll=False,
    time_major=False, reset_after=True, **kwargs
)
"""

#Columns for prediction
target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
    return np.transpose(
        np.array(
            df[cols]
            .applymap(lambda seq: [token2int[x] for x in seq])
            .values
            .tolist()
        ),
        (0, 2, 1)
    )

train_inputs = preprocess_inputs(train.loc[train.SN_filter == 1])
train_labels = np.array(train.loc[train.SN_filter == 1][target_cols].values.tolist()).transpose((0, 2, 1))

len(train_inputs)

len(train_labels)

def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=(0, 1))
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=-1)

"""Parameter Explanation:
Seq_len = equal to train seq length 107
pred_len = 
Dropout: helps prevent overfitting. Dropout layer randomly set input units to 0 with frequency of 0.5 at each step during training time.
"""

def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(L.GRU(hidden_dim, dropout=dropout, return_sequences=True))

def build_model(seq_len=107, pred_len=68, dropout=0.5, embed_dim=75, hidden_dim=128):
    inputs = L.Input(shape=(seq_len, 3))

    embed = L.Embedding(input_dim=len(token2int), output_dim=embed_dim)(inputs)
    reshaped = tf.reshape(
        embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))

    hidden = gru_layer(hidden_dim, dropout)(reshaped)
    hidden = gru_layer(hidden_dim, dropout)(hidden)
    hidden = gru_layer(hidden_dim, dropout)(hidden)
    
    # Since we are only making predictions on the first part of each sequence, we have
    # to truncate it
    truncated = hidden[:, :pred_len]
    
    out = L.Dense(5, activation='linear')(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    model.compile(tf.keras.optimizers.Adam(), loss=MCRMSE)
    
    return model

model = build_model(embed_dim=len(token2int))
model.summary()

x_train, x_val, y_train, y_val = train_test_split(
    train_inputs, train_labels, test_size=.2, random_state=42
)

history = model.fit(
    x_train, 
    y_train,
    validation_data=(x_val, y_val),
    batch_size=64,
    epochs=60,
    verbose=2,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(patience=5),
        tf.keras.callbacks.ModelCheckpoint('model.h5')
    ]
)

plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.legend(['val_loss', 'loss'], loc='upper right')

print(f"Min training loss={min(history.history['loss'])}, min validation loss={min(history.history['val_loss'])}")

"""## Make prediction on Test dataset both public and private

Public and private test datasets have different sequence lengths, therefore, we will preprocess each dataset separately and load the model using different tensor. 

* Note: for this project, since I am not submitting the project I will not generate the result for private prediction results.
"""

public_df = test.query("seq_length == 107")
private_df = test.query("seq_length == 130")

public_inputs = preprocess_inputs(public_df)
private_inputs = preprocess_inputs(private_df)

gru_public = build_model(seq_len=107,pred_len = 107,embed_dim=len(token2int))
gru_private = build_model(seq_len=130,pred_len = 130,embed_dim=len(token2int))

gru_public.load_weights('model.h5')
gru_private.load_weights('model.h5')

gru_public_preds = gru_public.predict(public_inputs)
gru_private_preds = gru_private.predict(private_inputs)

preds_ls = []

for df, preds in [(public_df, gru_public_preds), (private_df, gru_private_preds)]:
    for i, uid in enumerate(df.id):
        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=target_cols)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]
        preds_ls.append(single_df)

preds_df = pd.concat(preds_ls)

#Prediction dataframe:
preds_df

submission = sample_sub[['id_seqpos']].merge(preds_df, on=['id_seqpos'])
submission.to_csv('/content/drive/My Drive/submission.csv', index=False)

#Final prediction score on Kaggle: 0.28810
