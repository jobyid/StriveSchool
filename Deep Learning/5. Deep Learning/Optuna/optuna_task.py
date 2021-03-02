import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import optuna
from optuna.trial import TrialState
from sklearn.model_selection import train_test_split
batch_size = 20
cols = ['year', 'timbre_avg_0', 'timbre_avg_1', 'timbre_avg_2', 'timbre_avg_3',
       'timbre_avg_4', 'timbre_avg_5', 'timbre_avg_6', 'timbre_avg_7',
       'timbre_avg_8', 'timbre_avg_9', 'timbre_avg_10', 'timbre_avg_11',
       'timbre_cov_0', 'timbre_cov_1', 'timbre_cov_2', 'timbre_cov_3',
       'timbre_cov_4', 'timbre_cov_5', 'timbre_cov_6', 'timbre_cov_7',
       'timbre_cov_8', 'timbre_cov_9', 'timbre_cov_10', 'timbre_cov_11',
       'timbre_cov_12', 'timbre_cov_13', 'timbre_cov_14', 'timbre_cov_15',
       'timbre_cov_16', 'timbre_cov_17', 'timbre_cov_18', 'timbre_cov_19',
       'timbre_cov_20', 'timbre_cov_21', 'timbre_cov_22', 'timbre_cov_23',
       'timbre_cov_24', 'timbre_cov_25', 'timbre_cov_26', 'timbre_cov_27',
       'timbre_cov_28', 'timbre_cov_29', 'timbre_cov_30', 'timbre_cov_31',
       'timbre_cov_32', 'timbre_cov_33', 'timbre_cov_34', 'timbre_cov_35',
       'timbre_cov_36', 'timbre_cov_37', 'timbre_cov_38', 'timbre_cov_39',
       'timbre_cov_40', 'timbre_cov_41', 'timbre_cov_42', 'timbre_cov_43',
       'timbre_cov_44', 'timbre_cov_45', 'timbre_cov_46', 'timbre_cov_47',
       'timbre_cov_48', 'timbre_cov_49', 'timbre_cov_50', 'timbre_cov_51',
       'timbre_cov_52', 'timbre_cov_53', 'timbre_cov_54', 'timbre_cov_55',
       'timbre_cov_56', 'timbre_cov_57', 'timbre_cov_58', 'timbre_cov_59',
       'timbre_cov_60', 'timbre_cov_61', 'timbre_cov_62', 'timbre_cov_63',
       'timbre_cov_64', 'timbre_cov_65', 'timbre_cov_66', 'timbre_cov_67',
       'timbre_cov_68', 'timbre_cov_69', 'timbre_cov_70', 'timbre_cov_71',
       'timbre_cov_72', 'timbre_cov_73', 'timbre_cov_74', 'timbre_cov_75',
       'timbre_cov_76', 'timbre_cov_77']

class SongDataSet(torch.utils.data.Dataset):
    def __init__(self, file_dir):
        self.file = pd.DataFrame(file_dir, columns=cols)
        self.x = self.file.drop('year', axis=1)
        self.y = self.file.year
    def __len__(self):
        return len(self.x)
    def __getitem__(self, item):
        row   = self.x.iloc[item,:]
        label = self.y.iloc[item,0:1]

        sample = {"row":row, "label":label}
        return sample

train, val = train_test_split("songs.csv", test_size=0.2)

train_loader   = torch.utils.data.DataLoader(SongDataSet(train), batch_size = batch_size,shuffle = True)
test_loader    = torch.utils.data.DataLoader(SongDataSet(val), batch_size = batch_size, suffle = True)

def define_model(trial):
    n_layers = trial.suggest_int("n_layers", 1,3)
    layers = []
    in_features = 91
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i),4,150)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.1,0.7)
        layers.append(nn.Dropout(p))
        in_features = out_features
    layers.append(nn.Linear(in_features,1)) #output layer

    return nn.Sequential(*layers)

def objective(trial):
    pass




