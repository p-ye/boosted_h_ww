import pandas as pd
import awkward as ak
import uproot
from joblib import Parallel, delayed
import os
import numpy as np

all_df_names = ["signal_df"# "background_280_df", "background_500_df",
                # "background_1000_df"
                ]

for i in range(len(all_df_names)):
    df_name = all_df_names[i]
    df_fj = pd.read_csv(f"data/{df_name}_fj.csv")
    df_fjc = pd.read_csv(f"data/{df_name}_fjc.csv")
    df_fj['index'] = df_fj.groupby('event_index').cumcount()

    df_fjc.rename(columns={'ind': 'index'}, inplace=True)

    count_df = df_fjc.groupby(['event_index', 'index']).size().reset_index(name='count')

    df_fj_merged = pd.merge(
        df_fj,
        count_df,
        left_on=['event_index', 'index'],
        right_on=['event_index', 'index'],
        how='left'
    )
    df_fj_merged.to_csv(f"data/{df_name}_fj.csv")
    print("saved: "+df_name)
