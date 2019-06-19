import numpy as np
import pandas as pd

from tqdm import tqdm

import torch

from sklearn.model_selection import train_test_split

def getData(fileLocation = '~/Downloads/sudoku.csv'):
    data = pd.read_csv(fileLocation)

    dataSplit = []
    trueSplit = []

    # Format: flat starting grid (0 padded), flat end grid

    for d in tqdm(data.values[:2 ** 11]):
        dataSplit.append(list(map(int, list(d[0]))))
        trueSplit.append(list(map(int, list(d[1]))))

    dataSplit = np.array(dataSplit)
    trueSplit = np.array(trueSplit)

    return dataSplit, trueSplit

def preprocessData(batch_size = 1024, test_split = 0.1):

    # Shapes (N, 81), (N, 81)
    question, response = getData()

    n_batches = question.shape[0] // batch_size

    # Throw away the excess for the moment
    questionBatched = question[:n_batches * batch_size].reshape((n_batches, batch_size, 81))
    responseBatched = response[:n_batches * batch_size].reshape((n_batches, batch_size, 81))

    qTrain, qTest, rTrain, rTest = train_test_split(questionBatched, responseBatched, test_size = test_split)

    return torch.tensor(qTrain, dtype = torch.long), \
           torch.tensor(qTest, dtype = torch.long), \
           torch.tensor(rTrain, dtype = torch.long), \
           torch.tensor(rTest, dtype = torch.long)

if __name__ == '__main__':
    preprocessData()
