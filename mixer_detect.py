import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import sklearn.metrics
from warnings import filterwarnings
from sklearn.metrics import accuracy_score
from collections import defaultdict, deque
filterwarnings('ignore') 

# dev_set = []
# with open('./Dataset/dev_subset.jl', 'r') as f:
#     for l in tqdm(f):
#         d = json.loads(l)
#         dev_set.append((d[0][1:], d[1][1:], d[2]))

class TxDataSet(torch.utils.data.Dataset):
    def __init__(self, samples, max_len0=30, max_len1=10):
        self.samples = samples
        self.max_len0 = max_len0  
        self.max_len1 = max_len1
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        t0 = sample[0]
        t1 = sample[1]
        label = sample[2]
        l0 = len(t0)
        l1 = len(t1)
        if l0 < self.max_len0:
            t0 += [[0] * 78 for _ in range(self.max_len0 - l0)]
        if l1 < self.max_len1:
            t1 += [[0] * 78 for _ in range(self.max_len1 - l1)]
        return l0, t0, l1, t1, label, index

def collate_fx(batch):
    l0 = [b[0] for b in batch]
    t0 = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    l1 = [b[2] for b in batch]
    t1 = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    label = torch.tensor([b[4] for b in batch] , dtype=torch.float32)
    index = [b[5] for b in batch]
    return l0, t0, l1, t1, label, index

class DoubleLSTMClassify(nn.Module):
    def __init__(self, embedding_dim=78, hidden_dim=256):
        super(DoubleLSTMClassify, self).__init__()
        self.hidden_dim = hidden_dim
#         self.embedding = nn.Embedding(word_size, embedding_dim)
#         self.embedding.from_pretrained(torch.FloatTensor(vec))
        self.lstm0 = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=3, bidirectional=True, batch_first=True)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=3, bidirectional=True, batch_first=True)
        self.cls = nn.Linear(hidden_dim*2, 1)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, l0, t0, l1, t1):
        batch_size = t0.shape[0]
#         b1 = self.embedding(s1)
#         b2 = self.embedding(s2)
        k0 = self.lstm0(t0)[0]
        k1 = self.lstm1(t1)[0]
        x = torch.zeros(batch_size, self.hidden_dim*2).to(device)
        for i in range(batch_size):
            x[i][:self.hidden_dim] = k0[i][l0[i]-1]
            x[i][self.hidden_dim:] = k1[i][l1[i]-1]
        x = self.cls(x)
        x = self.sigmoid(x)
        return x
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_len0 = 30
max_len1 = 10
batch_size = 512
data_workers = 0

model = DoubleLSTMClassify(78)
model.to(device)

learning_rate = 1e-3
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
patience = 5
loss_list = []
epochs = 10
criterion = nn.BCELoss()
# train_dataset = TxDataSet(train_set)
# train_data_loader = torch.utils.data.DataLoader(
#     train_dataset,
#     batch_size=batch_size,
#     shuffle = True,
#     num_workers=data_workers,
#     collate_fn=collate_fx,
# )

# dev_dataset = TxDataSet(dev_set)
# dev_data_loader = torch.utils.data.DataLoader(
#     dev_dataset,
#     batch_size=batch_size,
#     shuffle = False,
#     num_workers=data_workers,
#     collate_fn=collate_fx,
# )

model.load_state_dict(torch.load('./mixer_lstm.pt',map_location='cpu', weights_only=False))

# Generate Precursor Tree 

tx_js = [
    {
    "transaction": [
      "bbefefd2f5da927a570213c8e98418ebdff4deebf48726ceabdd1ea58655f654"
    ],
    "precursors": [
      "0d267f19bde7f50365dc05ae7d2c03e5d9dd3e382619199d45b86640cae6cd8b"
    ]
  },
  {
    "transaction": [
      "0d267f19bde7f50365dc05ae7d2c03e5d9dd3e382619199d45b86640cae6cd8b"
    ],
    "precursors": [
      "691a7e28f76b88001c0a59b06bdd5759271e4e86d9cfb9cef338ce9c4cc3d0bd"
    ]
  },
  {
    "transaction": [
      "691a7e28f76b88001c0a59b06bdd5759271e4e86d9cfb9cef338ce9c4cc3d0bd"
    ],
    "precursors": [
      "1d0d08eb74d1b44b780b0571aec3b88b1541d00b5ded359901179c4f08f65446"
    ]
  },
  {
    "transaction": [
      "1d0d08eb74d1b44b780b0571aec3b88b1541d00b5ded359901179c4f08f65446"
    ],
    "precursors": [
      "012e93365369509f47520f52c2bb7970d6149f62403954fe3a8686453144016d",
      "39e855dedacce4c14570cea5a42618eb634442e7070f8e3688220dff72ceaba9",
      "3ad529407ccced341ceb930faa41f23f8ec52b8e8d348d271120cab92d587286",
      "5b66155bb209af8b20bbd1903635a6add78416adbeb1b81f640050ccb9936032",
      "6cda1fec2ad02b1e1a551224c075c9c74492d8122371df067fd2afbfb22cbbb4",
      "a48b08265de541b01bcfd342e37d438530d48321d045e1c89e31801d5632879e",
      "9f8f5252d41bd2117df685c5b84bc47a65c91d4956b509834fc3c458b9e46236",
      "eadfc1caf90f15f44c12658e971bd9f7c908e988e2a4ce83059edec2372135b8",
      "20e8b3e4b0b945de4691cfa0a94f0ef325d1f39631f5d665807a48cd0612e92f",
      "72b6f28ab5bda67b4fe001746c050d3af484002ac596ec586864d535119bd07f",
      "b768e9836b547cefd99a8982fb5413840989fa97e3303afedac83a6ee9f19114",
      "83d92f183353755777fc97794fae534c615b265d427e833cff55c1765080914d",
      "ee1e842534964e57880b1a7b004a91553693312c6852a9885006a3f37da5bd5a",
      "6510580f4b6b0c9aa9f70b123fbdb91a21d160a8ce523468d1c5cdaa268f9757",
      "b85d6089f2270cc394e70e3425c5ae86dc08781c41ac05226c5f49fb159e4fe7",
      "8a8703a721e0faebdcb43c3180e84cc4f6c66d0a1f85047d84c3cbd2f24e6a13",
      "22e7a5d390159014afc0855033e6abc85f325296c814fb587a1f06af48d9dac8",
      "d016fe27300966634165d4867efe92f9dab60dcd3bca477513a0d86bceb49570",
      "e2200252e8605602418ef36b077e7ecbcc9910cb0834e7a35162ce1b107fa007"
    ]
  }
]

# "transaction_hash":
# "input_amount_sum":
# "output_amount_sum": 
# "transaction_fee":
# "input_amount_std_dev":
# "output_amount_std_dev":
# "input_count_avg":
# "transaction_size":
# "avg_input_amount":
# "output_count_avg":
# "avg_output_amount":
# "input_address_count":
# "output_address_count":
# "transaction_weight":
# "lock_time":
# "is_coinbase":

tx_features = {
'bbefefd2f5da927a570213c8e98418ebdff4deebf48726ceabdd1ea58655f654':[5000000000, 5000000000, 0, 5000000000, 5000000000, 1, 0.0, 5000000000.0, 1, 4000000000, 1000000000, 2, 1500000000.0, 2500000000.0, 2],
'0d267f19bde7f50365dc05ae7d2c03e5d9dd3e382619199d45b86640cae6cd8b':[2500000000, 2500000000, 0, 2400000000, 100000000, 2, 1150000000.0, 1250000000.0, 1, 2400000000, 100000000, 2, 1150000000.0, 1250000000.0, 2],
'691a7e28f76b88001c0a59b06bdd5759271e4e86d9cfb9cef338ce9c4cc3d0bd':[5000000000, 5000000000, 0, 5000000000, 5000000000, 1, 0.0, 5000000000.0, 1, 2500000000, 2500000000, 2, 0.0, 2500000000.0, 2],
'1d0d08eb74d1b44b780b0571aec3b88b1541d00b5ded359901179c4f08f65446':[10000000000, 10000000000, 0, 5000000000, 5000000000, 2, 0.0, 5000000000.0, 2, 10000000000, 10000000000, 1, 0.0, 10000000000.0, 1],
'012e93365369509f47520f52c2bb7970d6149f62403954fe3a8686453144016d':[50000000000, 50000000000, 0, 5000000000, 5000000000, 10, 0.0, 5000000000.0, 10, 50000000000, 50000000000, 1, 0.0, 50000000000.0, 1],
'39e855dedacce4c14570cea5a42618eb634442e7070f8e3688220dff72ceaba9':[27500000000, 27500000000, 0, 5000000000, 2500000000, 6, 931694990.6249125, 4583333333.333333, 6, 27500000000, 27500000000, 1, 0.0, 27500000000.0, 1],
'3ad529407ccced341ceb930faa41f23f8ec52b8e8d348d271120cab92d587286':[50000000000, 50000000000, 0, 5000000000, 5000000000, 10, 0.0, 5000000000.0, 10, 50000000000, 50000000000, 1, 0.0, 50000000000.0, 1],
'5b66155bb209af8b20bbd1903635a6add78416adbeb1b81f640050ccb9936032':[27500000000, 27500000000, 0, 5000000000, 2500000000, 6, 931694990.6249125, 4583333333.333333, 6, 27500000000, 27500000000, 1, 0.0, 27500000000.0, 1],
'6cda1fec2ad02b1e1a551224c075c9c74492d8122371df067fd2afbfb22cbbb4':[5000000000, 5000000000, 0, 5000000000, 5000000000, 1, 0.0, 5000000000.0, 1, 4000000000, 1000000000, 2, 1500000000.0, 2500000000.0, 2],
'a48b08265de541b01bcfd342e37d438530d48321d045e1c89e31801d5632879e':[2500000000, 2500000000, 0, 2400000000, 100000000, 2, 1150000000.0, 1250000000.0, 1, 2400000000, 100000000, 2, 1150000000.0, 1250000000.0, 2],
'9f8f5252d41bd2117df685c5b84bc47a65c91d4956b509834fc3c458b9e46236':[5000000000, 5000000000, 0, 5000000000, 5000000000, 1, 0.0, 5000000000.0, 1, 2500000000, 2500000000, 2, 0.0, 2500000000.0, 2],
'eadfc1caf90f15f44c12658e971bd9f7c908e988e2a4ce83059edec2372135b8':[10000000000, 10000000000, 0, 5000000000, 5000000000, 2, 0.0, 5000000000.0, 2, 10000000000, 10000000000, 1, 0.0, 10000000000.0, 1],
'20e8b3e4b0b945de4691cfa0a94f0ef325d1f39631f5d665807a48cd0612e92f':[50000000000, 50000000000, 0, 5000000000, 5000000000, 10, 0.0, 5000000000.0, 10, 50000000000, 50000000000, 1, 0.0, 50000000000.0, 1],
'72b6f28ab5bda67b4fe001746c050d3af484002ac596ec586864d535119bd07f':[27500000000, 27500000000, 0, 5000000000, 2500000000, 6, 931694990.6249125, 4583333333.333333, 6, 27500000000, 27500000000, 1, 0.0, 27500000000.0, 1],
'b768e9836b547cefd99a8982fb5413840989fa97e3303afedac83a6ee9f19114':[50000000000, 50000000000, 0, 5000000000, 5000000000, 10, 0.0, 5000000000.0, 10, 50000000000, 50000000000, 1, 0.0, 50000000000.0, 1],
'83d92f183353755777fc97794fae534c615b265d427e833cff55c1765080914d':[27500000000, 27500000000, 0, 5000000000, 2500000000, 6, 931694990.6249125, 4583333333.333333, 6, 27500000000, 27500000000, 1, 0.0, 27500000000.0, 1],
'ee1e842534964e57880b1a7b004a91553693312c6852a9885006a3f37da5bd5a':[5000000000, 5000000000, 0, 5000000000, 5000000000, 1, 0.0, 5000000000.0, 1, 4000000000, 1000000000, 2, 1500000000.0, 2500000000.0, 2],
'6510580f4b6b0c9aa9f70b123fbdb91a21d160a8ce523468d1c5cdaa268f9757':[2500000000, 2500000000, 0, 2400000000, 100000000, 2, 1150000000.0, 1250000000.0, 1, 2400000000, 100000000, 2, 1150000000.0, 1250000000.0, 2],
'b85d6089f2270cc394e70e3425c5ae86dc08781c41ac05226c5f49fb159e4fe7':[5000000000, 5000000000, 0, 5000000000, 5000000000, 1, 0.0, 5000000000.0, 1, 2500000000, 2500000000, 2, 0.0, 2500000000.0, 2],
'8a8703a721e0faebdcb43c3180e84cc4f6c66d0a1f85047d84c3cbd2f24e6a13':[10000000000, 10000000000, 0, 5000000000, 5000000000, 2, 0.0, 5000000000.0, 2, 10000000000, 10000000000, 1, 0.0, 10000000000.0, 1],
'22e7a5d390159014afc0855033e6abc85f325296c814fb587a1f06af48d9dac8':[50000000000, 50000000000, 0, 5000000000, 5000000000, 10, 0.0, 5000000000.0, 10, 50000000000, 50000000000, 1, 0.0, 50000000000.0, 1],
'd016fe27300966634165d4867efe92f9dab60dcd3bca477513a0d86bceb49570':[27500000000, 27500000000, 0, 5000000000, 2500000000, 6, 931694990.6249125, 4583333333.333333, 6, 27500000000, 27500000000, 1, 0.0, 27500000000.0, 1],
'e2200252e8605602418ef36b077e7ecbcc9910cb0834e7a35162ce1b107fa007':[50000000000, 50000000000, 0, 5000000000, 5000000000, 10, 0.0, 5000000000.0, 10, 50000000000, 50000000000, 1, 0.0, 50000000000.0, 1]
}

tree = defaultdict(list)
for item in tx_js:
    tx = item['transaction'][0]
    for precursor in item['precursors']:
        tree[precursor].append(tx)

root = [k for k in tree.keys() if k not in [item["transaction"][0] for item in tx_js]][0]
queue = deque([(root, 0)])  # (transaction, level)
level_features = defaultdict(lambda: defaultdict(list))

while queue:
    node, level = queue.popleft()
    features = tx_features[node]
    for i, feature in enumerate(features):
        level_features[level][i].append(feature)
    for child in tree[node]:
        queue.append((child, level + 1))

level_stats_succ = defaultdict(list)
for level, feature_dict in level_features.items():
    # Add total transactions at this level
    total_transactions = len(feature_dict[0])  # All features will have the same number of entries
    level_stats = [total_transactions, 0, total_transactions]
    
    # Compute statistics for each feature
    for feature_idx, values in feature_dict.items():
        values = np.array(values)
        level_stats.extend([
            np.sum(values),
            np.max(values),
            np.min(values),
            np.std(values),
            np.mean(values)
        ])
    
    # Ensure we have 76 values: 1 (total) + 15 features * 5 stats
    assert len(level_stats) == 78
    level_stats_succ[level] = level_stats

stats_list = []
for i,j in level_stats_succ.items():
    stats_list.append(j)

# Generate Successor tree levels 

tx_js = [
  {
    "transaction": "549dddd80fb5203e63e057a2139ce9d00357f94b59c7ed8778c5e1f1aa953db9",
    "successors": [
      "4eb717c3e79d70dea15c2cb5cf8470f271244bea2dac7f9ec1789ad4feec4054",
      "8ea778e48f24f879408b3f0facbb49cd91d93b07119eba76f3c28a2efa65eceb",
      "a38d8f2466951dc96cf7bb6a27401b25ecba78b04b1bb063ab2237f11ae7af8b",
      "018a9b136b018779ff205584cc008105cd0c4f72f0e7d92acf2ecb1b5cd83a14"
    ]
  },
  {
    "transaction": "018a9b136b018779ff205584cc008105cd0c4f72f0e7d92acf2ecb1b5cd83a14",
    "successors": [
      "d2030383e96bfb12049ff2c3dead03d58863426dba8bd4a712aff4b95e062453"
    ]
  },
  {
    "transaction": "d2030383e96bfb12049ff2c3dead03d58863426dba8bd4a712aff4b95e062453",
    "successors": [
      "cab8d5baf4ab79ce917e6f4035e155add1a15cad0be8eb8e30d845ba838fcedf"
    ]
  }
]

tx_features = {
'549dddd80fb5203e63e057a2139ce9d00357f94b59c7ed8778c5e1f1aa953db9':[5000000000, 5000000000, 0, 5000000000, 5000000000, 1, 0.0, 5000000000.0, 1, 4000000000, 1000000000, 2, 1500000000.0, 2500000000.0, 2],
'4eb717c3e79d70dea15c2cb5cf8470f271244bea2dac7f9ec1789ad4feec4054':[2500000000, 2500000000, 0, 2400000000, 100000000, 2, 1150000000.0, 1250000000.0, 1, 2400000000, 100000000, 2, 1150000000.0, 1250000000.0, 2],
'8ea778e48f24f879408b3f0facbb49cd91d93b07119eba76f3c28a2efa65eceb':[5000000000, 5000000000, 0, 5000000000, 5000000000, 1, 0.0, 5000000000.0, 1, 2500000000, 2500000000, 2, 0.0, 2500000000.0, 2],
'a38d8f2466951dc96cf7bb6a27401b25ecba78b04b1bb063ab2237f11ae7af8b':[10000000000, 10000000000, 0, 5000000000, 5000000000, 2, 0.0, 5000000000.0, 2, 10000000000, 10000000000, 1, 0.0, 10000000000.0, 1],
'018a9b136b018779ff205584cc008105cd0c4f72f0e7d92acf2ecb1b5cd83a14':[50000000000, 50000000000, 0, 5000000000, 5000000000, 10, 0.0, 5000000000.0, 10, 50000000000, 50000000000, 1, 0.0, 50000000000.0, 1],
'd2030383e96bfb12049ff2c3dead03d58863426dba8bd4a712aff4b95e062453':[27500000000, 27500000000, 0, 5000000000, 2500000000, 6, 931694990.6249125, 4583333333.333333, 6, 27500000000, 27500000000, 1, 0.0, 27500000000.0, 1],
'cab8d5baf4ab79ce917e6f4035e155add1a15cad0be8eb8e30d845ba838fcedf':[50000000000, 50000000000, 0, 5000000000, 5000000000, 10, 0.0, 5000000000.0, 10, 50000000000, 50000000000, 1, 0.0, 50000000000.0, 1],
}

tree = defaultdict(list)
for item in tx_js:
    tx = item['transaction']
    for successor in item['successors']:
        tree[tx].append(successor)

all_successors = {s for item in tx_js for s in item["successors"]}
root = next(t for t in tree if t not in all_successors)
queue = deque([(root, 0)])  # (transaction, level)
level_features = defaultdict(lambda: defaultdict(list))

while queue:
    node, level = queue.popleft()
    features = tx_features[node]
    for i, feature in enumerate(features):
        level_features[level][i].append(feature)
    for child in tree[node]:
        queue.append((child, level + 1))

level_stats_succ = defaultdict(list)
for level, feature_dict in level_features.items():
    # Add total transactions at this level
    total_transactions = len(feature_dict[0])  # All features will have the same number of entries
    level_stats = [total_transactions, 0, total_transactions]
    
    # Compute statistics for each feature
    for feature_idx, values in feature_dict.items():
        values = np.array(values)
        level_stats.extend([
            np.sum(values),
            np.max(values),
            np.min(values),
            np.std(values),
            np.mean(values)
        ])
    
    # Ensure we have 76 values: 1 (total) + 15 features * 5 stats
    assert len(level_stats) == 78
    level_stats_succ[level] = level_stats

stats_list_succ = []
for i,j in level_stats_succ.items():
    stats_list_succ.append(j)

test_set = [(stats_list, stats_list_succ, 0)]
# print(level_stats)
test_dataset = TxDataSet(test_set)
for batch in test_dataset:
    l0, t0, l1, t1, label, index = batch
    t0 = torch.tensor(t0).unsqueeze(dim=0).to(device).type(torch.float32)
    t1 = torch.tensor(t1).unsqueeze(dim=0).to(device).type(torch.float32)
    label = torch.tensor(label).unsqueeze(dim=0).to(device).type(torch.float32)
    pred = model([l0], t0, [l1], t1)
    pred_labels = (pred > 0.5).int()

print(f'Prediction : {pred_labels}')