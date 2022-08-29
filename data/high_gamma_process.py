import os
import torch
import pickle


def load_highgamma_data_single_subject(filename, subject_id, to_tensor=True):
    # subject_id = str(subject_id)
    train_path = os.path.join(filename, 'high_gamma_train.pkl')
    test_path = os.path.join(filename, 'high_gamma_test.pkl')
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    train_X, train_y = train_data[subject_id]['X'], train_data[subject_id]['y']
    test_X, test_y = test_data[subject_id]['X'], test_data[subject_id]['y']
    if to_tensor:
        train_X = torch.tensor(train_X, dtype=torch.float32)
        test_X = torch.tensor(test_X, dtype=torch.float32)
        train_y = torch.tensor(train_y, dtype=torch.int64)
        test_y = torch.tensor(test_y, dtype=torch.int64)
    return train_X, train_y, test_X, test_y