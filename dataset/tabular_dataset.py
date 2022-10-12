from torch.utils.data import Dataset


class TabularDataset(Dataset):

    def __init__(self, X, y, budgets, learning_curves):
        self.X = X
        self.y = y
        self.budgets = budgets
        self.learning_curves = learning_curves

    def __len__(self):
        return self.y.size

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx], self.budgets[idx], self.learning_curves[idx]




