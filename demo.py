import os
import torch
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data.serialize import SerializerSettings
from models.utils import grid_iter
from models.promptcast import get_promptcast_predictions_data
from models.darts import get_arima_predictions_data
from models.llmtime import get_llmtime_predictions_data
from data.small_context import get_datasets
from models.validation_likelihood_tuning import get_autotuned_predictions_data

def plot_preds(train, test, pred_dict, model_name, show_samples=False):
    pred = pred_dict['median']
    pred = pd.Series(pred, index=test.index)
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(train)
    plt.plot(test, label='Truth', color='black')
    plt.plot(pred, label=model_name, color='purple')
    # shade 90% confidence interval
    samples = pred_dict['samples']
    lower = np.quantile(samples, 0.05, axis=0)
    upper = np.quantile(samples, 0.95, axis=0)
    plt.fill_between(pred.index, lower, upper, alpha=0.3, color='purple')
    if show_samples:
        samples = pred_dict['samples']
        samples = samples.values if isinstance(samples, pd.DataFrame) else samples
        for i in range(min(10, samples.shape[0])):
            plt.plot(pred.index, samples[i], color='purple', alpha=0.3, linewidth=1)
    plt.legend(loc='upper left')
    if 'NLL/D' in pred_dict:
        nll = pred_dict['NLL/D']
        if nll is not None:
            plt.text(0.03, 0.85, f'NLL/D: {nll:.2f}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.show()

print(torch.cuda.max_memory_allocated())
print()

llama_hypers = dict(
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True)
)

mistral_hypers = dict(
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True)
)

promptcast_hypers = dict(
    temp=0.7,
    settings=SerializerSettings(base=10, prec=0, signed=True, 
                                time_sep=', ',
                                bit_sep='',
                                plus_sign='',
                                minus_sign='-',
                                half_bin_correction=False,
                                decimal_point='')
)

arima_hypers = dict(p=[12,30], d=[1,2], q=[0])

model_hypers = {
     'LLMTime Llama-7B': {**llama_hypers},
     'LLMTime Llama-13B': {**llama_hypers},
     'LLMTime Llama-70B': {**llama_hypers},
     'PromptCast Llama-7B': {**promptcast_hypers},
     'mistral': {**mistral_hypers},
     'mistral-api-tiny': {**mistral_hypers},
     'mistral-api-small': {**mistral_hypers},
     'mistral-api-medium': {**mistral_hypers},
     'ARIMA': arima_hypers,
 }


def get_llama_predictions_data(train, test, model, **kwargs):
    kwargs.pop('model', None)  # Ensure 'model' is not passed to llama_completion_fn
    return get_llmtime_predictions_data(train, test, model=model, **kwargs)

model_predict_fns = {
    'LLMTime Llama-7B': lambda *args, **kwargs: get_llama_predictions_data(*args, model='llama-7b', **kwargs),
    'LLMTime Llama-13B': lambda *args, **kwargs: get_llama_predictions_data(*args, model='llama-13b', **kwargs),
    'LLMTime Llama-70B': lambda *args, **kwargs: get_llama_predictions_data(*args, model='llama-70b', **kwargs),
    'PromptCast Llama-7B': lambda *args, **kwargs: get_llama_predictions_data(*args, model='llama-7b', **kwargs),
    'mistral': lambda *args, **kwargs: get_llama_predictions_data(*args, model='mistral', **kwargs),
    'mistral-api-tiny': lambda *args, **kwargs: get_llama_predictions_data(*args, model='mistral-api-tiny', **kwargs),
}

model_names = list(model_predict_fns.keys())

datasets = get_datasets()
ds_name = 'AirPassengersDataset'

data = datasets[ds_name]
train, test = data 
out = {}

for model in model_names: 
    model_hypers[model].update({'dataset_name': ds_name})  
    hypers = list(grid_iter(model_hypers[model]))
    num_samples = 10
    pred_dict = get_autotuned_predictions_data(train, test, hypers, num_samples, model_predict_fns[model], verbose=False, parallel=False)
    out[model] = pred_dict
    plot_preds(train, test, pred_dict, model, show_samples=True)
