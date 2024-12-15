"""
This file provides utility functions for initializing and configuring RNA-based token prediction models 
from the multimolecule library to be used for fine-tuning. The file also allows easy setting up of models 
for testing or general usage, including tokenizer configuration and retrieval of maximum input sizes.

Functions:
- set_multimolecule_model: Loads a model and tokenizer based on the model name and configuration.
"""

from multimolecule import (
    RnaTokenizer,
    RnaFmForTokenPrediction,
    RnaFmConfig,
    RnaMsmForTokenPrediction,
    RnaMsmConfig,
    ErnieRnaForTokenPrediction,
    ErnieRnaConfig,
    UtrLmForTokenPrediction,
    UtrLmConfig,
    SpliceBertForTokenPrediction,
    SpliceBertConfig,
    RnaBertForTokenPrediction,
    RnaBertConfig
)
import pandas as pd

def set_multimolecule_model(model_name, for_testing = False):
    """
    Initializes and returns the model, tokenizer, and maximum input size based on the provided model name.

    Parameters:
    - model_name (str): The name of the model to be loaded.

    Returns:
    - tuple: a tuple containing the loaded and configured model, tokenizer, and the maximum input size.
    """

    # Mapping of model names to their respective token prediction models and max input sizes
    model_mapping = {
        "rnafm": {
            "model": RnaFmForTokenPrediction,
            "config": RnaFmConfig,
            "max_input_size": 1024,
        },
        "rnamsm": {
            "model": RnaMsmForTokenPrediction,
            "config": RnaMsmConfig,
            "max_input_size": 1022,
        },
        "ernierna": {
            "model": ErnieRnaForTokenPrediction,
            "config": ErnieRnaConfig,
            "max_input_size": 1024,
        },
        "utrlm-te_el": {
            "model": UtrLmForTokenPrediction,
            "config": UtrLmConfig,
            "max_input_size": 1024,
        },
        "splicebert": {
            "model": SpliceBertForTokenPrediction,
            "config": SpliceBertConfig,
            "max_input_size": 512,
        },
        "rnabert": {
            "model": RnaBertForTokenPrediction,
            "config": RnaBertConfig,
            "max_input_size": 438,
        }}

    if model_name not in model_mapping:
        raise ValueError(f"Unknown model name: '{model_name}'. Available models are: {', '.join(model_mapping.keys())}.")

    # Config definition
    config = model_mapping[model_name]["config"]()
    config.problem_type = "single_label_classification"
    config.num_labels = 2

    max_input_size = model_mapping[model_name]["max_input_size"]

    # Model and tokenizer definition
    if for_testing:
        loc_name = f"multimolecule-{model_name}-finetuned-secondary-structure/final_model"
        model = model_mapping[model_name]["model"].from_pretrained(loc_name, config=config)
        tokenizer = RnaTokenizer.from_pretrained(loc_name)
        
        #Load the ideal threashold
        df = pd.read_csv(f"multimolecule-{model_name}-finetuned-secondary-structure/eval_metrics.csv")
        best_threshold = df["eval_ideal_threshold"].iloc[-1]
        
        return model, tokenizer, max_input_size, best_threshold
        
    else:
        loc_name = f'multimolecule/{model_name}'
        model = model_mapping[model_name]["model"].from_pretrained(loc_name, config=config)
        tokenizer = RnaTokenizer.from_pretrained(loc_name)
        return model, tokenizer, max_input_size
