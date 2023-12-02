import numpy as np
from gradio import Progress
from src.modules.path_manager import LoadData, LoadModel, SaveModel
from src.modules.metrics import evaluate


class MatrixFactorization:
    def __init__(self, n_users: int, n_items: int, emb_dim: int):
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim

        self.user_emb = np.random.rand(n_users, emb_dim)
        self.item_emb = np.random.rand(n_items, emb_dim)

    def update_parameters(self, rank_matrix: np.ndarray, lr: float, reg: float = 0.02):
        for u in range(self.n_users):
            for i in range(self.n_items):
                if rank_matrix[u, i] > 0:
                    err = rank_matrix[u, i] - np.dot(self.user_emb[u], self.item_emb[i])
                    for dim in range(self.emb_dim):
                        self.user_emb[u, dim] += lr * (2 * err * self.item_emb[i, dim] - reg * self.user_emb[u, dim])
                        self.item_emb[i, dim] += lr * (2 * err * self.user_emb[u, dim] - reg * self.item_emb[i, dim])

    def predict(self, pairs: np.ndarray):
        predictions = np.zeros(shape=pairs.shape[0])
        for idx, (u, i) in enumerate(pairs):
            predictions[idx] = np.dot(self.user_emb[u], self.item_emb[i])
        return predictions


def train_and_evaluate(dataset: str, model_type: str, lr: float, reg: float, do_training: str, n_epochs: int, embedding_dim: int, output_model_name: str, progress: Progress = Progress()):
    # define saver and loader
    rank_loader = LoadData(dataset, 'rank-matrix')
    split_loader = LoadData(dataset, 'rank-split')
    checkpoint_loader = LoadModel('matrix_factorization')
    checkpoint_saver = SaveModel('matrix_factorization')

    # read dataset
    rank_matrix = rank_loader.read_dataframe('rank_matrix.csv', index_col=0, sep=',').to_numpy()
    metadata = rank_loader.read_json('metadata.json')
    x_test = split_loader.read_dataframe('x_test.csv', index_col=0, sep=',').to_numpy()
    y_test = split_loader.read_dataframe('y_test.csv', index_col=0, sep=',').to_numpy()

    # define model
    model = MatrixFactorization(metadata['n_users'], metadata['n_items'], embedding_dim)
    if model_type != 'random':
        user_emb, item_emb = checkpoint_loader.read_numpy(model_type, n_items=2)
        model.user_emb = user_emb
        model.item_emb = item_emb

    output_str = ''
    if do_training == 'train & evaluate':
        for epoch in progress.tqdm(range(n_epochs), desc='Training model'):
            model.update_parameters(rank_matrix, lr=lr, reg=reg)
            results = evaluate(model.predict(x_test), y_test)

            output_str += f'============ Epoch {epoch + 1}/{n_epochs} ============\n'
            for (metric, score) in results.items():
                output_str += f"{metric}: {score}\n"
            output_str += '\n'

        checkpoint_saver.save_numpy(model.user_emb, model.item_emb, file_name=output_model_name)
        output_str += '============ Save checkpoint ============\n'
        output_str += f'Saved checkpoint to: {output_model_name}\n'
    else:
        results = evaluate(model.predict(x_test), y_test)
        output_str += f'============ Evaluation ============\n'
        for (metric, score) in results.items():
            output_str += f"{metric}: {score}\n"
        output_str += '\n'

    return output_str





