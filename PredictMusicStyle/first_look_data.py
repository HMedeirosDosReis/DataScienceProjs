#!/usr/bin/env python3

# Henrique Medeiros Dos Reis
# 07/09/2024

import pandas as pd

pd.set_option('display.max_columns',None)
df = pd.read_csv('spotify_dataset.csv')
print(df.head())

print(df.shape)
