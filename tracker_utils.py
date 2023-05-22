## tracker utils
#Faizan: wrote code to identify the lost tracklet in real time, if it appears again.
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import ast
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import csv


##faizan added code for getting unique ids data
def unique_ids(df):
    """find unique ids in a data frame, these are the unique ids.
    
    Parameters
    ----------
    df: pd.DataFrame
        a dataframe of trackelts in the previous frames
    
    Return
    ------
    ids: list
         list of unique person ids in the previous frames
    """

    ids = list(set(df["id"].tolist()))
    return ids

def calculate_matrix_from_df(df):
    """make a  matrix from dataframe, roll all the features in column vector to create
    a big feature representation matrix that takes all the features from start to current frame
    
    
    Parameters
    ----------
    df: pd.DataFrame
        df of data of tracks
    
    Return
    ------
    matrix: np.ndarray
            rolled features into columns [x1 x2 x3 ... xm] where xi is a feature column vector associated with a detection"""
    
    #preprocess then df
    df["appearance"] = df["appearance"].apply(ast.literal_eval)
    matrix = np.column_stack(df["appearance"].values)
    return matrix

def cosine_similarity_measure(X, y):
    """calculate cosine similarity between column of X  and query vector  y
    
    Paramters
    ---------
    X: np.ndarray
       2d matrix with rows equal to number of features, and column is number of examples
       (nx, m)
    y: np.ndarray
       2d column vector
       (nx, 1)"""
    
    
    if y.shape != (y.shape[0], 1):
        y = y.reshape(-1, 1)

    dot_product = np.dot(X.T, y)
    
    magnitude_X = np.sqrt(np.sum(X**2, axis = 0, keepdims = True)).T
    magnitude_y = np.sqrt(np.sum(y**2))
    cosine_similarity = 1 - (dot_product / (magnitude_X * magnitude_y))
    
    return cosine_similarity


def calculate_similarity_score_for_all_data(features, query, threshold = 0.3):
    """calculate similarity scores between query and previous data stored in a matrix
    
    Parameters
    ----------
    features: np.ndarray
              feature matrix
    query: query track appearance to compare with
    
    Return
    ------
    min_value: float
    min_index: index of the minimum similarity distance appearance"""
    score_arr = cosine_similarity_measure(features, query)
    min_index = np.unravel_index(np.argmin(score_arr), score_arr.shape)
    min_value = score_arr[min_index]
    if min_value < threshold:
        index = min_index[0]
    else:
        index = None
    return (min_value,  index)

def find_match(features, query):
    """find best match using the data given in df and query comparison using cosine distance calculations
    
    Parameters
    ----------
    features: np.ndarray
        all previous appearances in one matrix
    
    query: np.nddary
           current track appearance to match"""

    min_value, min_index = calculate_similarity_score_for_all_data(features, query)
    return min_value, min_index


def latest_appearance(df, ids):
    """get a dataframe and return it's latest ids appearnce features
    
    Parameters
    ----------
    df: pd.DataFrame
        holding data about previous tracks such as id, cls, conf, frame, and appearance embeddings
    ids: list
        unique objects ids
    
    Return
    ------
    latest_ids_data: dict
        return unique ids that were present recently"""

    latest_ids_data = {}
    for id in ids:
        arr = df["id"].values == id
        index = max([i for i, x in enumerate(arr) if x])
        latest_ids_data[f"id: {id}"] = {"appearance": df["appearance"][index], "id": id, "frame": df["frame"][index]}
    return latest_ids_data

def compare_appearance(previous_tracklets, new_tracklet, similarity_threshold = 0.6):
    """compare two appearnce in current tracklet and in the prevoius ones
    
    Parameters
    ----------
    previous_tracklets: dict
        set of previous tracklets
    new_tracklet: dict
        a new tracklet
    
    Return
    ------
    assignment_id: int
        a unique id found by comparing features embeddings"""
    
    ids_scores = []
    new_track_appearance = new_tracklet["appearance"]
    new_track_appearance = ast.literal_eval(new_track_appearance)
    new_track_appearance = np.array(new_track_appearance, ndmin=2)
    for old_track in previous_tracklets.values():
        old_track_appearance = old_track["appearance"]
        old_track_appearance = ast.literal_eval(old_track_appearance)
        old_track_appearance = np.array(old_track_appearance, ndmin=2)
        dis = float(1 - cosine_similarity(old_track_appearance, new_track_appearance)[0][0])
        ids_scores.append((dis, old_track["id"]))

    best_similarity = min(ids_scores)[0]
    if best_similarity <= similarity_threshold:
        assignmet_id = min(ids_scores)[1]
        return assignmet_id
    else:
        assignmet_id = None
        return assignmet_id
    
def get_id(df, min_index):
    if min_index is not None:
        index = int(min_index)
        assigned_id = df.loc[index].id
    else:
        assigned_id = None
    return assigned_id
    
def generate_unique_id(ids):
    ids_set = set(ids)
    margin = 10
    max_id = max(ids) + margin
    while True:
        rand_id = random.randint(1, max_id)
        if rand_id not in ids_set:
            return rand_id
