"""Shared virtual-cell I/O helpers."""
import pickle


def load_pickle(path):
    with open(path, 'rb') as stream:
        return pickle.load(stream)
