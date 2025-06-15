import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

class DataSplitter:
    def __init__(
        self, 
        df: pd.DataFrame, 
        feature_col: str | list[str], 
        label_cols: str | list[str], 
        stratify_method: str = 'auto'
    ):
        """
        Initialize the DataSplitter.
        
        :param df: pandas DataFrame containing the data
        :param feature_col: str, name of the column containing features (e.g., 'image_name')
        :param label_cols: list of str or str, name(s) of the column(s) containing labels
        :param stratify_method: str, 'auto', 'binary', or 'multilabel'
        """
        self.df = df
        self.feature_col = [feature_col] if isinstance(feature_col, str) else feature_col
        self.label_cols = [label_cols] if isinstance(label_cols, str) else label_cols
        
        if stratify_method == 'auto':
            self.stratify_method = 'binary' if len(self.label_cols) == 1 else 'multilabel'
        else:
            self.stratify_method = stratify_method
        
        self.X = self.df[self.feature_col].values
        self.y = self.df[self.label_cols].values
        
        if self.stratify_method == 'binary':
            self.y = self.y.ravel()

    def split(
        self, 
        train_size: float = 0.8, 
        val_size: float = 0.1, 
        test_size: float = 0.1, 
        random_state: int = 42
    )->dict[str, pd.DataFrame | np.ndarray]:
        """
        Split the data into train, validation, and test sets.
        
        :param train_size: float, proportion of data to use for training
        :param val_size: float, proportion of data to use for validation
        :param test_size: float, proportion of data to use for testing
        :param random_state: int, random state for reproducibility
        :return: dict containing split DataFrames and numpy arrays
        """
        if abs(train_size + val_size + test_size - 1.0) > 1e-10:
            raise ValueError("train_size, val_size, and test_size must sum to 1")

        if self.stratify_method == 'binary':
            splitter = StratifiedShuffleSplit
        else:
            splitter = MultilabelStratifiedShuffleSplit

        # First split: separate test set
        first_split = splitter(n_splits=1, test_size=test_size, random_state=random_state)
        train_val_index, test_index = next(first_split.split(self.X, self.y))
        
        X_train_val, X_test = self.X[train_val_index], self.X[test_index]
        y_train_val, y_test = self.y[train_val_index], self.y[test_index]
        
        # Second split: separate train and validation sets
        val_ratio = val_size / (train_size + val_size)
        second_split = splitter(n_splits=1, test_size=val_ratio, random_state=random_state)
        train_index, val_index = next(second_split.split(X_train_val, y_train_val))
        
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        y_train, y_val = y_train_val[train_index], y_train_val[val_index]
        
        # Create DataFrames
        columns = self.feature_col + self.label_cols
        train_df = pd.DataFrame(np.column_stack((X_train, y_train)), columns=columns)
        val_df = pd.DataFrame(np.column_stack((X_val, y_val)), columns=columns)
        test_df = pd.DataFrame(np.column_stack((X_test, y_test)), columns=columns)
        
        return {
            'train_df': train_df, 'val_df': val_df, 'test_df': test_df,
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }

    def check_distribution(
        self, 
        split_data: dict[str, pd.DataFrame | np.ndarray]
    )->dict[str, dict[str, np.ndarray]]:
        """
        Check the distribution of labels in the split datasets.
        
        :param split_data: dict, output from the split method
        :return: dict containing distribution information
        """
        distributions = {}
        for split in ['train', 'val', 'test']:
            if self.stratify_method == 'binary':
                distributions[split] = np.mean(split_data[f'y_{split}'])
            else:
                distributions[split] = np.mean(split_data[f'y_{split}'], axis=0)
        
        differences = {
            'train_val': np.abs(distributions['train'] - distributions['val']),
            'train_test': np.abs(distributions['train'] - distributions['test']),
            'val_test': np.abs(distributions['val'] - distributions['test'])
        }
        
        return {'distributions': distributions, 'differences': differences}