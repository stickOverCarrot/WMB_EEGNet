import numpy as np
import os
import pickle
import torch
import argparse
from collections import OrderedDict
from braindecode.datasets.bcic_iv_2b import BCICompetition4Set2B
from braindecode.datautil.signalproc import bandpass_cnt, exponential_running_standardize
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
import logging
from braindecode.mne_ext.signalproc import mne_apply

log = logging.getLogger(__name__)
log.setLevel('INFO')
logging.basicConfig(level=logging.INFO)


def main(data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get file paths:
    train_data_paths, test_data_paths = get_paths_raw_data(data_dir)

    # Frequency filter low cut
    low_cut_hz = 0.5

    # Process and save data
    save_processed_datatset(train_data_paths, test_data_paths, output_dir,
                            low_cut_hz)


def get_paths_raw_data(data_dir):
    #   subject_ids: list of indices in range [1, ..., 9]
    subject_ids = [x for x in range(1, 10)]
    train_ids = [x for x in range(1, 4)]
    test_ids = [x for x in range(4, 6)]

    train_data_paths = [{'gdf': data_dir + f"/B0{subject_id}0{ids}T.gdf",
                         'mat': data_dir + f"/B0{subject_id}0{ids}T.mat"}
                        for subject_id in subject_ids for ids in train_ids]
    test_data_paths = [{'gdf': data_dir + f"/B0{subject_id}0{ids}E.gdf",
                        'mat': data_dir + f"/B0{subject_id}0{ids}E.mat"}
                       for subject_id in subject_ids for ids in test_ids]

    return train_data_paths, test_data_paths


def save_processed_datatset(train_filenames, test_filenames, output_dir,
                            low_cut_hz):
    train_data, test_data = {}, {}
    for train_filename in train_filenames:
        subject_id = train_filename['mat'].split('/')[-1][2:3]
        log.info("Processing data:"+train_filename['mat'].split('/')[-1]+"...")
        full_train_set = process_bbci_data(train_filename['gdf'],
                                           train_filename['mat'], low_cut_hz)
        if subject_id not in train_data.keys():
            train_data[subject_id] = {'X': [], 'y': []}
        train_data[subject_id]['X'].append(full_train_set.X)
        train_data[subject_id]['y'].append(full_train_set.y)
    for subject_id in train_data.keys():
        train_data[subject_id]['X'] = np.concatenate(train_data[subject_id]['X'], axis=0)
        train_data[subject_id]['y'] = np.concatenate(train_data[subject_id]['y'], axis=0)

    for test_filename in test_filenames:
        subject_id = test_filename['mat'].split('/')[-1][2:3]
        full_train_set = process_bbci_data(test_filename['gdf'],
                                           test_filename['mat'], low_cut_hz)
        if subject_id not in test_data.keys():
            test_data[subject_id] = {'X': [], 'y': []}
        test_data[subject_id]['X'].append(full_train_set.X)
        test_data[subject_id]['y'].append(full_train_set.y)
    for subject_id in train_data.keys():
        test_data[subject_id]['X'] = np.concatenate(test_data[subject_id]['X'], axis=0)
        test_data[subject_id]['y'] = np.concatenate(test_data[subject_id]['y'], axis=0)

    with open(os.path.join(output_dir, 'bciciv_2b_train.pkl'), 'wb') as f:
        pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(output_dir, 'bciciv_2b_test.pkl'), 'wb') as f:
        pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)
    

def process_bbci_data(filename, labels_filename, low_cut_hz):
    ival = [-500, 4000]
    # ival = [0, 4000]
    high_cut_hz = 100
    factor_new = 1e-3
    init_block_size = 1000

    loader = BCICompetition4Set2B(filename, labels_filename=labels_filename)
    # loader = BCICompetition4Set2A(filename)
    cnt = loader.load()

    # Preprocessing
    cnt = cnt.pick(['EEG:C3', 'EEG:Cz', 'EEG:C4'])
    assert len(cnt.ch_names) == 3

    # lets convert to millvolt for numerical stability of next operations
    cnt = mne_apply(lambda a: a * 1e6, cnt)
    # cnt = mne_apply(lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz,
    #                                        cnt.info['sfreq'],
    #                                        filt_order=3,
    #                                        axis=1), cnt)
    # cnt = mne_apply(lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
    #                                                           init_block_size=init_block_size, eps=1e-4).T, cnt)

    marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],)])

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    return dataset


def preprocess(data_path, output_dir):
    train_data = {}
    test_data = {}

    for subject_id in range(1, 10):
        data = []
        label = []
        for i in range(1, 4):
            sub_path = "B0{}0{}T".format(subject_id, i)
            data_file = os.path.join(data_path, sub_path + "_data_raw.npy")
            label_file = os.path.join(data_path, sub_path + "_label.npy")
            data.append(np.load(data_file))
            label.append(np.load(label_file).argmax(axis=-1))
        train_data[subject_id] = {'X': np.concatenate(data, axis=0).transpose([0, 2, 1]), 'y': np.concatenate(label, axis=0)}
        data = []
        label = []
        for i in range(4, 6):
            sub_path = "B0{}0{}E".format(subject_id, i)
            data_file = os.path.join(data_path, sub_path + "_data_raw.npy")
            label_file = os.path.join(data_path, sub_path + "_label.npy")
            data.append(np.load(data_file))
            label.append(np.load(label_file).argmax(axis=-1))
        test_data[subject_id] = {'X': np.concatenate(data, axis=0).transpose([0, 2, 1]), 'y': np.concatenate(label, axis=0)}
    with open(os.path.join(output_dir, 'bciciv_2b_train.pkl'), 'wb') as f:
        pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(output_dir, 'bciciv_2b_test.pkl'), 'wb') as f:
        pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)


def load_bciciv2b_data_single_subject(filename, subject_id, to_tensor=True):
    subject_id = str(subject_id)
    train_path = os.path.join(filename, 'bciciv_2b_train.pkl')
    test_path = os.path.join(filename, 'bciciv_2b_test.pkl')
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    try:
        train_X, train_y = train_data[subject_id]['X'], train_data[subject_id]['y']
        test_X, test_y = test_data[subject_id]['X'], test_data[subject_id]['y']
    except KeyError:
        subject_id = int(subject_id)
        train_X, train_y = train_data[subject_id]['X'], train_data[subject_id]['y']
        test_X, test_y = test_data[subject_id]['X'], test_data[subject_id]['y']
    if to_tensor:
        train_X = torch.tensor(train_X, dtype=torch.float32)
        test_X = torch.tensor(test_X, dtype=torch.float32)
        train_y = torch.tensor(train_y, dtype=torch.int64)
        test_y = torch.tensor(test_y, dtype=torch.int64)
    return train_X, train_y, test_X, test_y


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Gdf to pkl")
    parser.add_argument('-data_path', type=str,
                        help='Gdf path to load')
    parser.add_argument('-output_path', type=str,
                        help='Pkl path to save')
    args_ = parser.parse_args()
    main(args_.data_path, args_.output_path)

