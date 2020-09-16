# OpenVaccine: mRNA Degradation Predictor

To win the fight against the COVID-19 pandemic, we require an effective vaccine that can be equitably and widely distributed. 
With 54 different types of vaccines under development, two of which are already entering the human trials, according to the World Health Organization. Among the different vaccine candidates is a new player called - mRNA vaccines. 
mRNA vaccine is developed by US pharmacy company Moderna and it began its human trials on March 16 2020. 

## About the mRNA Vaccine

Vaccines work by training the body to recognise and respond to the proteins produced by disease-causing organisms, such as a virus or bacteria. [1] Traditional vaccines are made up of small or inactivated does of the disease-causing organism, or the proteins that it produces, which can provoke our body’s immune system into mounting a response. 

However, mRNA vaccines are the total opposite, tricking the body into producing the viral proteins itself. The vaccines work by using mRNA, which is the molecule that puts DNA instructions into action. Inside a cell, mRNA is used as a template to build a protein. ‘An mRNA is basically like a pre-form of a protein and its (sequence encodes) what the protein is made of later on,’ said by Prof. Bekeredjian-Ding.

With these characteristics, mRNA vaccines combine desirable immunological properties with an outstanding safety profile and the unmet flexibility of genetic vaccines. Also because anytype of protein can be expressed from mRNA without the need to adjust the production process, mRNA has the maximum flexibility with respect to development.

## The issues with mRNA Vaccine

mRNA vaccines have taken the lead as the fastest vaccine candidates for COVID-19, but currently, they face key potential limitations. One of the biggest challenges is how to design a super stable mRNA. mRNA molecules have a high tendency to spontaneously degrade.

Currently, little is known on the details of where in the backbone of a given RNA is most prone to being affected. Without enough knowledge, mRNA vaccines can’t be packaged in disposable syringes and shipped under refrigeration around the world. 

## Project Goals

I am looking to develop the Deep-learning model (Gated Recurrent Unit) to predict likely degradation rates at each base of an RNA molecule, trained on a subset of an Eterna dataset comprising over 3000 RNA molecules (which span a panoply of sequences and structures) and their degradation rates at each position. 

The model is scored on the second generation of RNA sequences that have just been devised by Eterna players for COVID-19 mRNA vaccines.

Step-by-step instructions to build a GRU model for predicting mRNA degradation rate:
* *Exploratory data analysis* on train and test datasets. Extract information on the mRNA sequence, RNA structure, and signal to noise distribution. 
* *Preprocess inputs* for train dataset sequence, structure, and predicted loop type.
* Train the dataset on *GRU prediction model*. Train prediction info: **Min training loss=0.2548, min validation loss=0.2513**
* Apply *pre-train model on the public (Sequence: 107) and private (Sequence: 130) test dataset* and generate the prediction results on Kaggle. Test prediction info: Final prediction score on Kaggle: **0.2881**

**Click here for project files:**

**Exploratory data analysis** : [EDA Information](https://github.com/NaeRong/OpenVaccine-mRNA-Degradation-Predictor/blob/master/EDA/Data_info)

**GRU model:** [GRU Model Information](https://github.com/NaeRong/OpenVaccine-mRNA-Degradation-Predictor/blob/master/GRU%20model/Model_info)

Kaggle Competition link: [Kaggle OpenVaccine](https://www.kaggle.com/c/stanford-covid-vaccine)
<p align="center">
  <img src="https://github.com/NaeRong/OpenVaccine-mRNA-Degradation-Predictor/blob/master/Pictures/Sponsor.png">
</p>

