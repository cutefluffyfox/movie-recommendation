import numpy as np
import pandas as pd
from scipy import sparse
from gradio import Progress
from sklearn.linear_model import BayesianRidge
from src.modules.metrics import evaluate
from src.modules.path_manager import LoadData, LoadModel, SaveModel


class BayesianRidgeRegressor:
    def __init__(self, *args, **kwargs):
        self.model = BayesianRidge(*args, **kwargs)

    def train(self, x_train: sparse.csr_matrix, y_train: np.ndarray):
        self.model.fit(x_train.toarray(), y_train)

    def predict(self, x):
        return self.model.predict(x)


def train_and_evaluate(dataset: str, model_type: str, max_iter: int, tol: float, alpha_1: float, alpha_2: float, lambda_1: float, lambda_2: float, do_training: str, output_model_name: str, progress: Progress = Progress()):
    # define saver and loader
    feature_loader = LoadData(dataset, 'feature-split')
    model_loader = LoadModel('bayesian_ridge')
    model_saver = SaveModel('bayesian_ridge')

    # read dataset
    x_train = feature_loader.read_sparse('X_train.npz')
    y_train = feature_loader.read_dataframe('y_train.csv', sep=',').to_numpy().squeeze()
    x_test = feature_loader.read_sparse('X_test.npz')
    y_test = feature_loader.read_dataframe('y_test.csv', sep=',').to_numpy().squeeze()

    # define model
    model = BayesianRidgeRegressor(max_iter=max_iter, tol=tol, alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2)
    if model_type != 'random':
        model = model_loader.read_sklearn(model_type)

    for _ in progress.tqdm(range(int(do_training == 'train & evaluate')), desc='Training model'):  # gradio require iterator
        model.train(x_train, y_train)

    for _ in progress.tqdm(range(1), desc='Testing model'):
        predictions = model.predict(x_test)
        results = evaluate(predictions, y_test)

    output_str = f'============ Evaluation ============\n'
    for (metric, score) in results.items():
        output_str += f"{metric}: {score}\n"
    output_str += '\n'

    if do_training == 'train & evaluate':
        model_saver.save_sklearn(model, output_model_name)

    return output_str

