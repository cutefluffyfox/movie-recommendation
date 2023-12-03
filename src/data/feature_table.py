import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from src.modules.path_manager import SaveData, LoadData


numerical_preprocessing = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_preprocessing = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('one-hot', OneHotEncoder(handle_unknown='ignore'))
])


def preprocess_data(X: pd.DataFrame, y: pd.DataFrame, train_size: float, seed: int) -> tuple[sparse.csr_matrix, pd.Series, sparse.csr_matrix, pd.Series]:
    # get columns
    ignore_columns = [
        'unknown', 'action', 'adventure', 'animation', 'childrens', 'comedy',
        'crime', 'documentary', 'drama', 'fantasy', 'film_noir', 'horror',
        'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western'
    ]
    numerical_features = list(set(X.select_dtypes(include='number').columns) - set(ignore_columns))
    categorical_features = X.select_dtypes(exclude='number').columns

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_preprocessing, numerical_features),
        ('cat', categorical_preprocessing, categorical_features),
    ])

    # split and transform
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=seed)
    x_train = preprocessor.fit_transform(x_train, y_train)
    x_test = preprocessor.transform(x_test)

    return x_train, y_train, x_test, y_test


def split_data(dataset: str, train_size: float, seed: int) -> str:
    # define saver and loader
    loader = LoadData(dataset, dir_type='raw')
    saver = SaveData(dataset, 'feature-split')

    # read dataset
    rating_df = loader.read_dataframe('u.data', ignore_header=True, names=['user_id', 'item_id', 'rating', 'timestamp'])
    users_df = loader.read_dataframe(
        'u.user', ignore_header=True, sep='|', index_col=0,
        names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
    )
    movies_df = loader.read_dataframe(
        'u.item', ignore_header=True, encoding='latin-1', sep='|', index_col=0,
        names=[
           'item_id', 'title', 'release_date', 'video_release_date', 'url',
           'unknown', 'action', 'adventure', 'animation', 'childrens', 'comedy', 'crime', 'documentary', 'drama',
           'fantasy',
           'film_noir', 'horror', 'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western'
        ]
    )

    # join all tables into one big
    raw_total_df = pd.merge(rating_df, movies_df, on='item_id', how='inner')
    raw_total_df = pd.merge(raw_total_df, users_df, on='user_id', how='inner')

    # remove fully-unique columns and columns with no information
    total_df = raw_total_df.drop(columns=['user_id', 'item_id', 'video_release_date', 'url', 'timestamp'])

    # split to X, y
    x = total_df.drop(columns=['rating'])
    y = total_df.rating

    # preprocess and split
    x_train, y_train, x_test, y_test = preprocess_data(x, y, train_size=train_size, seed=seed)

    # save train
    saver.save_sparse(x_train, 'X_train.npz')
    saver.save_dataframe(y_train, 'y_train.csv', index=False)

    # save test
    saver.save_sparse(x_test, 'X_test.npz')
    saver.save_dataframe(y_test, 'y_test.csv', index=False)

    return 'Successfully split and preprocessed data'
