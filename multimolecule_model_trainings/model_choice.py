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

def set_multimolecule_model(model_name):
    """
    Initializes and returns the model, tokenizer, and maximum input size based on the provided model name.

    Parameters:
    - model_name (str): The name of the model to loaded.

    Returns:
    - tuple: a tuple containing the loaded and configured model, tokenizer, and the maximum input size.
    """

    # Mapping of model names to their respective classes and max input sizes
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

    #Model and tokenizer definition
    model = model_mapping[model_name]["model"].from_pretrained(f'multimolecule/{model_name}', config=config)
    tokenizer = RnaTokenizer.from_pretrained(f'multimolecule/{model_name}')
    max_input_size = model_mapping[model_name]["max_input_size"]

    return model, tokenizer, max_input_size