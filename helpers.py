import pandas as pd
import torch as pt

def category2codes(df: pd.DataFrame, column_name: str):
    new_column = df[column_name].astype('category')
    new_column = new_column.cat.codes
    return new_column


class MyNet(pt.nn.Module):
    
    def __init__(self, input, output):
        super(MyNet, self).__init__()
        self.layer1 = pt.nn.Linear(input.shape[1], 512) # addapt to whatever input we want to use this with ;)
        self.layer2 = pt.nn.Dropout(0.3)
        self.layer3 = pt.nn.Linear(512, output.shape[1]) # addapt to whatever input we want to use this with ;)
        self.layer4 = pt.nn.ReLU()
    
    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features