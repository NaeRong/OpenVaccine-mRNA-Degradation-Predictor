
# GRU Model to predict likely degradation rates at RNA molecule
mRNA subcellular localization mechanisms play an important role in post-transcriptional gene regulation.
Zipcodes are the cis-regulatory elements from a different RNA-binding proteins interating with 
**Next**

## What is RNA degradation rate?
## Why using Deep learning framework works?
## What is GRU model?  
## Model building process
* Pre-processing columns and define target columns
According to the competition description, only the following 5 columns are required for our model. 


target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
```
We also need to preprocess the column inputs before putting into the model. The encoding process is showed as below:
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
Text preprocessing is an important step for NLP models. It transforms text into a more readable form so that machine learning algorithms can perform better.
* Model framework
* Model evaluation 
* Model optimization 





