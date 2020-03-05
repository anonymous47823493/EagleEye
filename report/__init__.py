import distiller
import os
import os.path as osp
import pandas as pd
import numpy as np

def weights_sparsity_summary(model, opt=None):
    try:
        df = distiller.weights_sparsity_summary(model.module, return_total_sparsity=True)
    except AttributeError:
        df = distiller.weights_sparsity_summary(model, return_total_sparsity=True)
    return df[0]['NNZ (dense)'].sum() // 2

def performance_summary(model, dummy_input, opt=None, prefix=""):
    try:
        df = distiller.model_performance_summary(model.module, dummy_input)
    except AttributeError:
        df = distiller.model_performance_summary(model, dummy_input)
    new_entry = {
        'Name':['Total'],
        'MACs':[df['MACs'].sum()],
    }
    MAC_total = df['MACs'].sum()
    return MAC_total

def model_summary(model, dummy_input, opt=None):
    return performance_summary(model, dummy_input, opt), weights_sparsity_summary(model, opt)