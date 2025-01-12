# LSTM Mixer Detection for Cryptocurrency Transactions

## Overview

This repository contains the implementation of **LSTM-TC (LSTM Transaction Classifier)**, a novel deep learning-based method for detecting Bitcoin coin mixing transactions. Coin mixing techniques enhance transaction privacy by obscuring address linkages, making detection crucial for accurate blockchain analysis and Bitcoin address clustering.

It surpasses traditional rule-based and graph neural network-based approaches in recall and precision, enabling the detection of new and evolving coin mixing patterns. Along with changes to the base model & the training, I have also added a script to convert transactions into the required input & output transactio tree. The original paper does not provide any information on the features , they have used & Hence , the features I am using are estimates based on experimentation.  

The dataset & paper are available at : 
- [Google Drive](https://drive.google.com/drive/folders/1srpyBEXbaDhLg5juEQh-I71IxUA3JYx1) 
- [Original paper](https://link.springer.com/article/10.1007/s10489-021-02453-9)

---

## Key Features

- **High Recall Detection**: Effective identification of coin mixing transactions, reducing false negatives.
- **Deep Learning Approach**: Utilizes an LSTM-based classifier that leverages transaction trees for feature extraction and classification.
- **Real-Time Processing**: Optimized for large-scale Bitcoin blockchain data with rapid transaction analysis.
- **Transaction Tree Extractor**: Creates pre & post cursor tree based on transaction data which are inputs to the LSTM model.
- **Dataset Included**: Comprehensive labeled datasets for training and testing, accessible in this repository.

---
## Project Structure
```
.
├── mixer_lstm.pt  # PyTorch trained model weights 
├── mixer_detect.py # Code to load mixer model & use it against transactions declared in an array.
├── requirements.txt # Python package requirements
├── model.ipynb # Contains the base model code as well as Custom Dataset , Model Training , Evaluation & Inference methods.
└── README.md  # This file
```

## Methodology

The LSTM-TC workflow involves three main steps:

1. **Transaction Tree Extraction**:
   - Precursor (N-level) and successor (M-level) transactions are traced to construct transaction trees.

2. **Tree Serialization**:
   - Each transaction tree layer is aggregated into fixed-length vectors using statistical features like input/output sums, transaction counts, and processing fees.

3. **Classification**:
   - Serialized sequences are passed through an LSTM-based model, which outputs a classification score indicating whether a transaction is a coin mixing instance.

These are the 15 bitcoin features which are used : 

- transaction_hash:
- input_amount_sum:
- output_amount_sum: 
- transaction_fee:
- input_amount_std_dev:
- output_amount_std_dev:
- input_count_avg:
- transaction_size:
- avg_input_amount:
- output_count_avg:
- avg_output_amount:
- input_address_count:
- output_address_count:
- transaction_weight:
- lock_time:
- is_coinbase:

These 15 features are aggregated [ sum, max, min, std, mean ] level wise from the transaction tree to get 78 features for each level to get the processed pre-cursor & post-cursor tree which can be passed as input to the model.

---


## Results

### Performance Metrics

| **Model**      | **Precision** | **Recall** | **F1-Score** | **ROC-AUC-Score** |
|-----------------|---------------|------------|--------------|--------------|
| LSTM-TC        | 0.966         | 0.961      | 0.964        | 0.964        |
| Rule-Based     | 0.821         | 0.650      | 0.722        | 0.964        |
| GCN Classifier | 0.579         | 0.654      | 0.614        | 0.964        |

Following are the Results achieved by me on the Dev Dataset after training on 500 epochs with a subset[25%] of original training data: 
 - Avg Accuracy: 0.8992745535714286
 - Avg Precision: 0.9155580885733609
 - Avg Recall: 0.8804509101539322
 - Avg F1 score: 0.8973077397184309
 - Avg ROC-AUC score: 0.8994153705695987


## Installation

### Prerequisites
- Python >= 3.8
- PyTorch >= 1.7
- Scikit-learn >= 0.24
- GPU support (optional but recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/bitcoin-lstm-tc.git
   cd bitcoin-lstm-tc
2. Run mixer_detect.py :
   ```bash
   python mixer_detect.py

