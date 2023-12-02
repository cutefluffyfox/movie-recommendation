import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split
from src.modules.path_manager import SaveData, LoadData


def split_data(dataset: str, train_size: float, seed: int) -> str:
    # define saver and loader
    loader = LoadData(dataset, dir_type='raw')
    saver = SaveData(dataset, 'rank-split')

    # read dataset
    rating_df = loader.read_dataframe('u.data', ignore_header=True, names=['user_id', 'item_id', 'rating', 'timestamp'])

    # get number of users and items
    n_users = int(rating_df.user_id.max())
    n_items = int(rating_df.item_id.max())

    # make id's from 0 (not from 1)
    rating_df.user_id -= 1
    rating_df.item_id -= 1

    # split data
    x_train, x_test, y_train, y_test = train_test_split(
        rating_df[['user_id', 'item_id']],
        rating_df.rating,
        train_size=train_size,
        random_state=seed
    )

    # check whether all users present in the dataset
    if len(x_train.user_id.unique()) != n_users:
        gr.Warning('Number of users is not the same as was expected, may result in future problems')

    # save train
    saver.save_dataframe(x_train, 'X_train.csv')
    saver.save_dataframe(y_train, 'y_train.csv')

    # save test
    saver.save_dataframe(x_test, 'X_test.csv')
    saver.save_dataframe(y_test, 'y_test.csv')

    # save additional info about dataset
    saver.save_json({'n_users': n_users, 'n_items': n_items}, 'metadata.json')

    return 'Successfully split data'


def make_rank_matrix(dataset: str) -> str:
    # define saver and loader
    loader = LoadData(dataset, 'rank-split')
    saver = SaveData(dataset, 'rank-matrix')

    # read dataset
    X_train = loader.read_dataframe('X_train.csv', index_col=0, sep=',')
    y_train = loader.read_dataframe('y_train.csv', index_col=0, sep=',')
    metadata = loader.read_json('metadata.json')

    # transform dataset and make rank matrix
    df_train = pd.concat([X_train, y_train], axis=1)
    df_train = df_train.reset_index()

    rank_matrix = df_train.pivot(index="user_id", columns="item_id", values="rating").fillna(0)

    # add columns (items) that was skipped
    for skipped_item in set(range(metadata['n_items'])) - set(rank_matrix.columns):
        rank_matrix[skipped_item] = 0

    # rearrange columns in right order
    rank_matrix = rank_matrix[list(range(metadata['n_items']))]

    # save rank matrix
    saver.save_dataframe(rank_matrix, 'rank_matrix.csv')
    saver.save_json(metadata, 'metadata.json')

    return 'Successfully made rank matrix'
