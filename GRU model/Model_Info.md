
# GRU Model to predict likely degradation rates at RNA molecule
mRNA subcellular localization mechanisms play an important role in post-transcriptional gene regulation.
Zipcodes are the cis-regulatory elements from a different RNA-binding proteins interating with 

## RNN - Deep Learning Model 
### Tokenize our RNA sequences before feeding into the GRU model:

As the competition required, the **target columns (Columns to predict)** are
* 'reactivity'
* 'deg_Mg_pH10'
* 'deg_pH10'
* 'deg_Mg_50C'
* 'deg_50C'

Tokenization process is shown as below. 

Text preprocessing is an important step for NLP models. It transforms text into a more readable form so that machine learning algorithms can perform better.

#### 1. Define token to int function. Since mRNA only contains "ACGUBEHIMSX" characters, we are assigning an index for each character.  

```python
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}
print(token2int)
>>>{'(': 0, ')': 1, '.': 2, 'A': 3, 'C': 4, 'G': 5, 'U': 6, 'B': 7, 'E': 8, 'H': 9, 'I': 10, 'M': 11, 'S': 12, 'X': 13}
```
#### 2. Below is the Tokenization function. We are only transforming the "sequence', 'structure', 'predicted_loop_type' columns. 
```txt
Function definition:
numpy.transpose(a, axes=None)
inputs:
    *a: input
    *axes:If specified, it must be a tuple or list which contains a permutation of [0,1,..,N-1] where N is the number of axes of a. The i’th axis of the returned array will correspond to the axis numbered axes[i] of the input.
```
Applying token2int function to all the sequence and assining the output matrix to (0,2,1) axes.
```python
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
```
Applying preprocess_inputs function to training inputs and labels. Also setting train dataset SN_filter to 1, since we have observed that SN_filter == 1 values in test dataset.
```python
train_inputs = preprocess_inputs(train.loc[train.SN_filter == 1])
train_labels = np.array(train.loc[train.SN_filter == 1][target_cols].values.tolist()).transpose((0, 2, 1))
```
```python
len(train_inputs)
>>> 1589
len(train_labels)
>>> 1589
```
### Model Evaluation 
According to the competition guideline, Mean Columnwise Root Mean Squared Error(𝑀𝐶𝑅𝑀𝑆𝐸) is the model evaluation formula.

𝑀𝐶𝑅𝑀𝑆𝐸 = 1𝑚∑𝑚𝑗=1𝑅𝑀𝑆𝐸𝑗

* 𝑚  - number of predicted variables

* 𝑛 - number of test samples

* 𝑦𝑖𝑗 - 𝑖-th actual value of 𝑗​-th variable

* 𝑦𝑖𝑗 - 𝑖-th predicted value of 𝑗-th variable

The MCRMSE is simply an average across all RMSE values for each of our columns. It can allow us to use a single-number evaluation metric, even in the case of multiple outputs.

```python
def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=(0, 1))
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=-1)
```
### GRU Model Framwork

A Gated Recurrent Unit (GRU) model is a modification to the RNN hidden layer that makes it much better capturing long-range connections and better navigating the vanishing gradient problems. 

The GRU operates using a reset gate and an update gate. The reset gate is between the prior activation and the next candidate activation to forget the previous state. The update gate will then decide how much of the candidate activation to use in updating the cell state.

Both LSTMs and GRUs can keep the memory from previous activations rather than replacing the entire activation like a traditional RNN. However, 
while the GRU performs both of forget and input gates together via its reset gate, LSTM separates these operations.

With different architectures, GRU can train faster and less gradient decent volatility rate - Might be caused by fewer gates for the gradients to flow through so more steady progress after many epochs - than the LSTM.

```python
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
```
We applied the build_model function on the token2int list and displayed the model detailed information.

```python
model = build_model(embed_dim=len(token2int))
model.summary()
```
<p align="center">
  <img src="https://github.com/NaeRong/OpenVaccine-mRNA-Degradation-Predictor/blob/master/Pictures/Model_Info.png">
</p>

Defined train and validation datasets with an 8 and 2 ratio.

```python
x_train, x_val, y_train, y_val = train_test_split(
    train_inputs, train_labels, test_size=.2, random_state=42
)
```
Trained the model on train and validation datasets. In this model, I used the callback function to:

* tf.keras.callbacks.ReduceLROnPlateau: to reduce the learning rate when a metric has stopped improving. 

Patience is the number of epochs with no improvement after which the learning rate will be reduced. In this case, I will stop the model after 5 epochs without improvement. 

* tf.keras.callbacks.ModelCheckpoint: Callback to save the Keras model or model weights at some frequency. I saved our model under the name "model.h5"


*Note A callback is an object that can perform actions at various stages of training

```python 
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
```
Prediction result (Train dataset):
```python 
plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.legend(['val_loss', 'loss'], loc='upper right')
```
As we can see in the loss function graph, 57 epoch has the best model performance.

<p align="center">
  <img src="https://github.com/NaeRong/OpenVaccine-mRNA-Degradation-Predictor/blob/master/Pictures/val_loss_loss.png">
</p>


### Test Dataset Prediction(both public and private)

Public and private test datasets have different sequence lengths, therefore, we will preprocess each dataset separately and load the model using different tensor. 

Preprocessed test dataset under different sequence length (Public: 107 and Private 130), and load model weights from 'model.h5' we saved in the previous step.

```python
public_df = test.query("seq_length == 107")
private_df = test.query("seq_length == 130")

public_inputs = preprocess_inputs(public_df)
private_inputs = preprocess_inputs(private_df)

gru_public = build_model(seq_len=107,pred_len = 107,embed_dim=len(token2int))
gru_private = build_model(seq_len=130,pred_len = 130,embed_dim=len(token2int))

gru_public.load_weights('model.h5')
gru_private.load_weights('model.h5')
```

Applied GRU model on the test datasets to obtain predictions.
```Python
gru_public_preds = gru_public.predict(public_inputs)
gru_private_preds = gru_private.predict(private_inputs)
```

### Final Submission Result:
MCRMSE : 0.28810







