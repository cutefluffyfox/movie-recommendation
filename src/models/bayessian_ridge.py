import numpy as np
from gradio import Progress
from sklearn.linear_model import BayesianRidge
from src.modules.metrics import evaluate
from src.modules.path_manager import SaveData, LoadData



class BayesianRidgeRegressor:
    def __init__(self):
        self.model = BayesianRidge()

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(x_train.toarray(), y_train)

    def predict(self, x):
        return self.model.predict(x)


def train_and_evaluate(dataset: str, model_type: str, progress: Progress):
    # define saver and loader
    feature_loader = LoadData(dataset, 'feature-split')

    # read dataset
    x_train = feature_loader.read_dataframe('x_train.csv')
    y_train = feature_loader.read_dataframe('y_train.csv')
    x_test = feature_loader.read_dataframe('x_test.csv')
    y_test = feature_loader.read_dataframe('y_test.csv')

    # define model
    model = BayesianRidgeRegressor()
    # if model_type == 'random':
    #     model = MatrixFactorization(metadata['n_users'], metadata['n_items'], embedding_dim)
    # else:  # TODO: MAKE READING OF MODEL
    #     pass

    for _ in progress.tqdm(range(1), desc='Training model'):  # gradio require iterator
        model.train(x_train, y_train)

    for _ in progress.tqdm(range(1), desc='Testing model'):
        predictions = model.predict(x_test)
        results = evaluate(predictions, y_test)

    output_str = f'============ Evaluation ============\n'
    for (metric, score) in results.items():
        output_str += f"{metric}: {score}\n"
    output_str += '\n'

    return output_str

