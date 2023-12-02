import os
import ast
import json
import joblib
import numpy as np
import pandas as pd
from scipy import sparse


class Loader:
    """
    Base class for any Loader
    """
    dir = os.getcwd()

    def get_all_data(self):
        """
        returns list of all files in self.dir (main) directory
        """
        return os.listdir(self.dir)


class Saver:
    """
    Base class for any Saver
    """
    dir = os.getcwd()


class SaveData(Saver):
    """
    Save pandas DataFrames, torchtext Vocabs
    """
    def __init__(self, dataset: str, *extras, dir_type: str = 'intermediate'):
        # save each part
        self.dataset = dataset
        self.dir_type = dir_type
        self.extras = extras

        # combine into one main path
        self.dir = os.path.join(os.getcwd(), 'data', dir_type, dataset, *extras)

        # make directory if not exist
        os.makedirs(self.dir, exist_ok=True)

    def save_dataframe(self, csv: pd.DataFrame or pd.Series, file_name: str, index: bool = True) -> str:
        file_path = os.path.join(self.dir, file_name)
        csv.to_csv(file_path, index=index)
        return file_path

    def save_json(self, dict_data: dict, file_name: str) -> str:
        file_path = os.path.join(self.dir, file_name)
        with open(file_path, 'w') as file:
            json.dump(dict_data, file, indent=4)
        return file_path

    def save_sparse(self, sparse_data: sparse.csr_matrix, file_name: str) -> str:
        file_path = os.path.join(self.dir, file_name)
        sparse.save_npz(file_path, sparse_data)
        return file_path


class LoadData(Loader):
    """
    Load pandas DataFrames, torchtext Vocabs
    """
    def __init__(self, dataset: str, *extras, dir_type: str = 'intermediate'):
        # save each part of a path
        self.dataset = dataset
        self.dir_type = dir_type
        self.extras = extras

        # combine parts into one path
        self.dir = os.path.join(os.getcwd(), 'data', dir_type, dataset, *extras)

    def read_dataframe(
            self, file_name: str, apply_ast_to: str = None, ignore_header: bool = False,
            names: list[str] = None, sep: str = None, encoding: str = None, index_col: int = None) -> pd.DataFrame or pd.Series:

        # define key parameters to read dataframe
        kwargs = dict()
        if ignore_header:
            kwargs['header'] = None
        if names is not None:
            kwargs['names'] = names
        if sep is not None:
            kwargs['sep'] = sep
        if encoding is not None:
            kwargs['encoding'] = encoding
        if index_col is not None:
            kwargs['index_col'] = index_col

        # read dataframe
        df = pd.read_table(os.path.join(self.dir, file_name), **kwargs)

        # if some column supposed to have list type but is string, parse it
        if apply_ast_to is not None:
            df[apply_ast_to] = df[apply_ast_to].apply(ast.literal_eval)

        # return dataframe
        return df

    def read_json(self, file_name) -> dict:
        with open(os.path.join(self.dir, file_name), 'r') as file:
            dict_data = json.load(file)
        return dict_data

    def read_sparse(self, file_name: str) -> sparse.csr_matrix:
        return sparse.load_npz(os.path.join(self.dir, file_name))


class SaveModel(Loader):
    """
    Save numpy/sklearn models
    """
    def __init__(self, model_type: str, *extras, dir_type: str = 'checkpoints'):
        # save each part of a path
        self.model_type = model_type
        self.extras = extras
        self.dir_type = dir_type

        # combine parts into one path
        self.dir = os.path.join(os.getcwd(), dir_type, model_type, *extras)

        # make directory if not exist
        os.makedirs(self.dir, exist_ok=True)

    def save_numpy(self, *args, file_name: str) -> str:
        file_path = os.path.join(self.dir, file_name)
        with open(file_path, 'wb') as file:
            for weights in args:
                np.save(file, weights)
        return file_path

    def save_sklearn(self, model, file_name: str) -> str:
        file_path = os.path.join(self.dir, file_name)
        joblib.dump(model, file_path)
        return file_path


class LoadModel(Loader):
    """
    Load numpy/sklearn models
    """
    def __init__(self, model_type: str, *extras, dir_type: str = 'checkpoints'):
        # save each part of a path
        self.model_type = model_type
        self.extras = extras

        # combine parts into one path
        self.dir = os.path.join(os.getcwd(), dir_type, model_type, *extras)

    def read_numpy(self, file_name: str, n_items: int) -> list[np.ndarray]:
        with open(os.path.join(self.dir, file_name), 'rb') as file:
            items = [np.load(file) for _ in range(n_items)]
        return items

    def read_sklearn(self, file_name: str):
        return joblib.load(os.path.join(self.dir, file_name))


class GradioReaders:
    """
    Main class for all path's Gradio may require
    """
    @staticmethod
    def read_dir_type(dir_type: str, *extras):
        full_path = os.path.join(os.getcwd(), 'data', dir_type, *extras)
        return os.listdir(full_path) if os.path.exists(full_path) else []

    @staticmethod
    def vocab_readers(dataset: str):
        full_path = os.path.join(os.getcwd(), 'data', 'intermediate', dataset, 'vocab')
        return os.listdir(full_path) if os.path.exists(full_path) else []

    @staticmethod
    def checkpoint_readers(model_type: str, dir_type: str = 'checkpoints'):
        full_path = os.path.join(os.getcwd(), dir_type, model_type)
        return os.listdir(full_path) if os.path.exists(full_path) else []