import torch
import torch.util.data

def _train(train_lmdb_path):
    batch_size = 128

    model = Model()
    train_loader = torch.utils.data.DataLoader(Dataset(train_lmdb_path), batch_size=batch_size,
						shuffle=false)
    #evaluator = Evaluator()

