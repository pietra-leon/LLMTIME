import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from data.small_context import get_datasets
from data.serialize import SerializerSettings
from models.validation_likelihood_tuning import get_autotuned_predictions_data
from models.utils import grid_iter
from models.gaussian_process import get_gp_predictions_data
from models.darts import get_TCN_predictions_data, get_NHITS_predictions_data, get_NBEATS_predictions_data
from models.llmtime import get_llmtime_predictions_data
from models.darts import get_arima_predictions_data
import openai

openai.api_key = os.environ['OPENAI_API_KEY']
openai.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

# Specify the hyperparameter grid for each model
gpt3_hypers = dict(
    temp=.7,
    alpha=[0.5, .7, 0.9, 0.99],
    beta=[0, .15, 0.3, .5],
    basic=[False],
    settings=[SerializerSettings(base=10, prec=prec, signed=True, half_bin_correction=True) for prec in [2,3]],
)

gpt4_hypers = dict(
    alpha=0.3,
    basic=True,
    temp=1.0,
    top_p=0.8,
    settings=SerializerSettings(base=10, prec=3, signed=True, time_sep=', ', bit_sep='', minus_sign='-')
)

llama_hypers = dict(
    temp=1.0,
    alpha=0.99,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, time_sep=',', bit_sep='', plus_sign='', minus_sign='-', signed=True), 
)

deepseek_hypers = dict(
    temp=0.9,
    alpha=0.99,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, time_sep=',', bit_sep='', plus_sign='', minus_sign='-', signed=True), 
)

promptcast_hypers = dict(
    temp=.7,
    settings=SerializerSettings(base=10, prec=0, signed=True, 
                                time_sep=', ',
                                bit_sep='',
                                plus_sign='',
                                minus_sign='-',
                                half_bin_correction=False,
                                decimal_point='')
)

gp_hypers = dict(lr=[5e-3, 1e-2, 5e-2, 1e-1])

arima_hypers = dict(p=[12,20,30], d=[1,2], q=[0,1,2])

TCN_hypers = dict(in_len=[10, 100, 400], out_len=[1],
    kernel_size=[3, 5], num_filters=[1, 3], 
    likelihood=['laplace', 'gaussian']
)

NHITS_hypers = dict(in_len=[10, 100, 400], out_len=[1],
    layer_widths=[64, 16], num_layers=[1, 2], 
    likelihood=['laplace', 'gaussian']
)

NBEATS_hypers = dict(in_len=[10, 100, 400], out_len=[1],
    layer_widths=[64, 16], num_layers=[1, 2], 
    likelihood=['laplace', 'gaussian']
)

model_hypers = {
    'gp': gp_hypers,
    'arima': arima_hypers,
    'TCN': TCN_hypers,
    'N-BEATS': NBEATS_hypers,
    'N-HiTS': NHITS_hypers,
    'text-davinci-003': {'model': 'text-davinci-003', **gpt3_hypers},
    'gpt-4': {'model': 'gpt-4', **gpt4_hypers},
    'llama-70b': {'model': 'llama-70b', **llama_hypers},
    'deepseek-6.7b': {'model': 'deepseek-6.7b', **deepseek_hypers},  # Added DeepSeek-6.7B
    'deepseek-67b': {'model': 'deepseek-67b', **deepseek_hypers},  # Added DeepSeek-67B
    'deepseek-r1': {'model': 'deepseek-r1', **deepseek_hypers},  # Added DeepSeek-R1
}

# Specify the function to get predictions for each model
model_predict_fns = {
    'gp': get_gp_predictions_data,
    'arima': get_arima_predictions_data,
    'TCN': get_TCN_predictions_data,
    'N-BEATS': get_NBEATS_predictions_data,
    'N-HiTS': get_NHITS_predictions_data,
    'text-davinci-003': get_llmtime_predictions_data,
    'gpt-4': get_llmtime_predictions_data,
    'llama-70b': get_llmtime_predictions_data,
    'deepseek-6.7b': get_llmtime_predictions_data,  # Added DeepSeek
    'deepseek-67b': get_llmtime_predictions_data,  # Added DeepSeek
    'deepseek-r1': get_llmtime_predictions_data,  # Added DeepSeek
}

def is_gpt(model):
    """Checks if the model is a GPT-based model or DeepSeek."""
    return any([x in model for x in ['ada', 'babbage', 'curie', 'davinci', 'text-davinci-003', 'gpt-4', 'deepseek']])

# Specify the output directory for saving results
output_dir = 'outputs/darts'
os.makedirs(output_dir, exist_ok=True)

datasets = get_datasets()
for dsname, data in datasets.items():
    train, test = data
    if os.path.exists(f'{output_dir}/{dsname}.pkl'):
        with open(f'{output_dir}/{dsname}.pkl', 'rb') as f:
            out_dict = pickle.load(f)
    else:
        out_dict = {}

    # N-HiTS, TCN and N-BEATS require training and can be slow. Skip them if you want quick results.
    
    for model in ['text-davinci-003', 'gpt-4', 'gp', 'arima', 'N-HiTS', 'TCN', 'N-BEATS', 'deepseek-6.7b', 'deepseek-67b', 'deepseek-r1']:
        if model in out_dict:
            print(f"Skipping {dsname} {model}")
            continue
        else:
            print(f"Starting {dsname} {model}")
            hypers = list(grid_iter(model_hypers[model]))

        parallel = True if is_gpt(model) else False
        num_samples = 20 if is_gpt(model) else 100
        try:
            preds = get_autotuned_predictions_data(train, test, hypers, num_samples, model_predict_fns[model], verbose=False, parallel=parallel)
            out_dict[model] = preds
        except Exception as e:
            print(f"Failed {dsname} {model}")
            print(e)
            continue
        with open(f'{output_dir}/{dsname}.pkl', 'wb') as f:
            pickle.dump(out_dict, f)

    print(f"Finished {dsname}")