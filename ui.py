import gradio as gr

from src.modules.path_manager import GradioReaders
from src.models import bayessian_ridge, matrix_factorization
from src.data import feature_table, rank_matrix


def update_dropdown(func):
    def update_func_dropdown(*args, **kwargs):
        return gr.update(choices=func())
    return update_func_dropdown


def update_raw_dropdown(*args, **kwargs):
    new_choices = gr.update(choices=GradioReaders.read_dir_type('raw'))
    return new_choices


def update_checkpoint_dropdown(*args, **kwargs):
    checkpoint_choices = gr.update(choices=['random'] + GradioReaders.checkpoint_readers('matrix_factorization'))
    return checkpoint_choices


# make tabs
with gr.Blocks(theme=gr.themes.Default()) as ui:
    gr.Markdown('Movie recommendation models - Made by @cutefluffyfox')

    # matrix factorization tab
    with gr.Tab('Matrix factorization'):
        gr.Markdown('Split dataset & Factorize')
        with gr.Row():
            with gr.Column():
                raw_dataset = gr.Dropdown(GradioReaders.read_dir_type('raw'), label="Chose dataset", info="Chose what dataset to preprocess", interactive=True)
                update_raw_btn = gr.Button('Refresh dataset list')
                train_size = gr.Slider(0, 1, value=0.8, label='Train size', info='Ratio to train/test dataset')
                seed = gr.Number(value=42, label='Set seed (nothing - random seed)', info='Seed to reproduce results, do not enter for random seed', precision=0)
            with gr.Column():
                split_factor_output = gr.Text(label='Information box')

        split_btn = gr.Button('Split dataset', variant='primary')
        split_btn.click(rank_matrix.split_data, inputs=[raw_dataset, train_size, seed], outputs=[split_factor_output])

        factorize_btn = gr.Button('Make rank matrix', variant='primary')
        factorize_btn.click(rank_matrix.make_rank_matrix, inputs=[raw_dataset], outputs=split_factor_output)

        update_raw_btn.click(update_raw_dropdown, inputs=[], outputs=[raw_dataset])


        gr.Markdown('Train & Evaluate model')
        with gr.Row():
            with gr.Column():
                factorize_model = gr.Dropdown(['random'] + GradioReaders.checkpoint_readers('matrix_factorization'), value='random', label='Chose model', info='Chose model to train/evaluate', interactive=True)
                update_factorize_btn = gr.Button('Refresh model list')
                lr = gr.Slider(0, 1, value=6e-4, label='Learning rate', info='Learning rate (alpha)')
                regularization = gr.Slider(0, 1, value=0.02, label='Regularization', info='Regularization parameter')
                do_training = gr.Radio(['train & evaluate', 'evaluate'], value='train & evaluate', label='Chose whether to train model', info='Decide whether train model or just evaluate', show_label=True)
                n_epochs = gr.Number(value=40, label='Number of epochs to train model', info='Number of epochs to train model (used only if training is set)', precision=0)
                embedding_dim = gr.Number(value=8, label='Embedding dimension (used only in random model)', info='Embedding dimension for vectors', precision=0)
                model_name = gr.Text(value='model.npy', label='Name of output model checkpoint', info='Filename of output model checkpoint (used only if training is set)')
            with gr.Column():
                factorize_output = gr.Text(label='Training and information box')

        train_evaluate_factorize = gr.Button('Train & Evaluate', variant='primary')
        train_evaluate_factorize.click(matrix_factorization.train_and_evaluate, inputs=[raw_dataset, factorize_model, lr, regularization, do_training, n_epochs, embedding_dim, model_name], outputs=[factorize_output])

        update_factorize_btn.click(update_checkpoint_dropdown, inputs=[], outputs=[factorize_model])


ui.queue()
if __name__ == '__main__':
    ui.launch(inbrowser=True, show_error=True)
