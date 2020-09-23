# Exploratory Data Analysis on Train / Test / Sample Datasets

In this competition, you will be predicting the degradation rates at various locations along RNA sequence.
There are multiple ground truth values provided in the training data. While the submission format requires all 5 to be predicted, only the following are scored: **reactivity, deg_Mg_pH10, and deg_Mg_50C**
. Also, the mRNA sequence contains only these characters: **ACGUBEHIMSX**

- [Import Packages](#import-packages)
- [Load Zip Files](#load-zip-files)
- [Explore Shape And Null Values](#explore-shape-and-null-values)
- [Explore mRNA Sequence](#explore-mrna-sequence)
- [Explore mRNA Structure Column](#explore-mrna-structure-column)
- [Signal to Noise Feature](#signal-to-noise-feature)

## Import Packages
   ```python
    import pandas as pd, numpy as np
    import math, json, gc, random, os, sys
    from matplotlib import pyplot as plt
    import seaborn as sns
    import tensorflow as tf
    import tensorflow.keras.layers as L
    from sklearn.model_selection import train_test_split
   ```
## Load Zip Files 
Files include:
* train.json - the training data
* test.json - the test set, without any columns associated with the ground truth.
* sample_submission.csv - a sample submission file in the correct format

   ```python
   zf = zipfile.ZipFile('/content/drive/My Drive/Colab Notebooks/sample_submission.csv.zip') 
   zf_1 = zipfile.ZipFile('/content/drive/My Drive/Colab Notebooks/test.json.zip') 
   zf_2 = zipfile.ZipFile('/content/drive/My Drive/Colab Notebooks/train.json.zip') 
   sample_sub = pd.read_csv(zf.open('sample_submission.csv'))
   test = pd.read_json(zf_1.open('test.json'), lines=True)
   train = pd.read_json(zf_2.open('train.json'), lines=True)
   ```
## Explore Shape And Null Values

**Train dataset**:
* Row: 2400
* Col: 19
* No missing values
* Col names: ['index' 'id' 'sequence' 'structure' 'predicted_loop_type''signal_to_noise' 'SN_filter' 'seq_length' 'seq_scored' 'reactivity_error' 'deg_error_Mg_pH10' 'deg_error_pH10'
 'deg_error_Mg_50C' 'deg_error_50C' 'reactivity' 'deg_Mg_pH10' 'deg_pH10''deg_Mg_50C' 'deg_50C']
 
**Test dataset**:
* Row: 3634
* Col: 7
* No missing values
* Col names: ['index' 'id' 'sequence' 'structure' 'predicted_loop_type' 'seq_length''seq_scored']

**Sample dataset**:
* Row: 457953
* Col: 6
* No missing values
* Col names: ['id_seqpos' 'reactivity' 'deg_Mg_pH10' 'deg_pH10' 'deg_Mg_50C' 'deg_50C']
<p align="center">
  <img src="https://github.com/NaeRong/OpenVaccine-mRNA-Degradation-Predictor/blob/master/Pictures/Sample.png">
</p>

## Explore mRNA Sequence
sequence - (1x107 string in Train and Public Test, 130 in Private Test) Describes the RNA sequence, a combination of A, G, U, and C for each sample. Should be 107 characters long, and the first 68 bases should correspond to the 68 positions specified in seq_scored (note: indexed starting at 0).

Sequence length for train is {107} and for test is {130,107}. This information will be useful for defining the GRU model sequence length.
Train dataset sequence length value count is **Frequency: Counter({'A': 45, 'C': 23, 'U': 20, 'G': 19})**
```python
from collections import Counter
print(train.seq_length.value_counts(),'\nFrequency:',Counter(train.head(1).sequence.values[0]))
```
Then, we want to see the sequence legth in public test set:
```python 
sns.countplot(test['seq_length'].value_counts())
ax.set_title('Sequence Length in test set')
```
<p align="center">
  <img src="https://github.com/NaeRong/OpenVaccine-mRNA-Degradation-Predictor/blob/master/Pictures/Seq_len_test.png">
</p>

## Explore mRNA Structure Column
structure - (1x107 string in Train and Public Test, 130 in Private Test) An array of (, ), and . characters that describe whether a base is estimated to be paired or unpaired. Paired bases are denoted by opening and closing parentheses e.g. (....) means that base 0 is paired to base 5, and bases 1-4 are unpaired.

Train dataset structure:
```python
train.structure.head(1).values[0]
>> .....((((((.......)))).)).((.....((..((((((....))))))..)).....))....(((((((....))))))).....................
Counter(train.structure.head(1).values[0])
>> Counter({'(': 23, ')': 23, '.': 61})
```
Test dataset structure:
```python
test.structure.value_counts()
>> ......((((((((((.(((((.....))))))))((((((((...)))))...)))))))...))).(((((((....))))))).....................
```
## Signal to Noise Feature
Signal_to_noise and SN_filter columns control the 'quality' of samples! 

As per the data tab of this competition the samples in test.json (Both private and public test datasets) have been filtered in the following way:
1. Minimum value across all 5 conditions must be greater than -0.5.
2. Mean signal/noise across all 5 conditions must be greater than 1.0. [Signal/noise is defined as mean( measurement value over 68 nts )/mean( statistical error in measurement value over 68 nts)]
3. To help ensure sequence diversity, the resulting sequences were clustered into clusters with less than 50% sequence similarity, and the 629 test set sequences were chosen from clusters with 3 or fewer members. That is, any sequence in the test set should be sequence similar to at most 2 other sequences.


The signal to noise feature on train dataset has a distribution of a right-skewed. 
<p align="center">
  <img src="https://github.com/NaeRong/OpenVaccine-mRNA-Degradation-Predictor/blob/master/Pictures/Signal_Noise.png">
</p>
The signal to noise feature counts:
<p align="center">
  <img src="https://github.com/NaeRong/OpenVaccine-mRNA-Degradation-Predictor/blob/master/Pictures/Sig_Noise_cnt.png">
</p>

Important findings in the Signal to noise / SN_filer features are:
* Samples with signal_to_noise greater than 1: 2096
* Samples with SN_filter = 1: 1589
* Samples with signal_to_noise greater than 1, but SN_filter == 0: 509




