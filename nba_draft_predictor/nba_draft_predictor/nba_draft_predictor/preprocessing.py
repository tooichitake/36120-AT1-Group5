import pandas as pd

def merge_metadata(data_df, meta_df):
    return data_df.merge(meta_df, on="player_id", how="left")

def fill_missing(df):
    return df.fillna(0)
