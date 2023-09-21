import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch import nn
import matplotlib.pyplot as plt

class Classification(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 15)
        self.layer2 = nn.Linear(15, 1)
 
    def forward(self, x):
        x = self.layer2(self.layer1(x))
        return x

train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
features = ["Sex", "Age", "Survived"]
train_df = train_df[features]

train_df = pd.get_dummies(train_df).sample(frac = 1).dropna()
train_split = int(0.8*len(train_df))
test_df = train_df.iloc[train_split:,:]
train_df = train_df.iloc[:train_split,:]
x_train_df, y_train_df = train_df.drop(['Survived', 'Sex_male'] , axis = 1), train_df['Survived']
x_test_df, y_test_df = test_df.drop(['Survived', 'Sex_male'] , axis = 1), test_df['Survived']

X_train, y_train = torch.Tensor(x_train_df.values.astype('float32')), torch.Tensor(y_train_df.values.astype('float32'))
X_test, y_test = torch.Tensor(x_test_df.values.astype('float32')), torch.Tensor(y_test_df.values.astype('float32'))

model_1 = Classification()

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.01)

if "Survived" in features: 
    features.remove("Survived")
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
test_d = pd.get_dummies(test_df[features]).sample(frac = 1).drop(['Sex_male'], axis = 1)
for x in test_d.keys():
    mean = test_d[x].mean()
    test_d[x].fillna(mean, inplace=True)
model_1.eval()
test = torch.tensor(test_d.values.astype('float32'))
with torch.inference_mode():
    test_logits = model_1(test).squeeze()
    test_preds = torch.round(torch.sigmoid(test_logits))
output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': test_preds})
