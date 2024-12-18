from datasets import Dataset
from .data_preprocessing import *

def create_dataset(df, tokenizer, model, max_model_input_size, seed, sample_n_datapoints=None, truncate=True, shuffle=True):
    """
    Creates tokenized datasets for training, validation, and testing a model. 

    This function:
    1. Calculates branch point positions within intron sequences.
    2. Removes sequences that exceed the model's maximum input size.
    3. Splits the data into train, validation, and test sets based on predefined chromosome groups.
    4. Extracts intron sequences and their corresponding labels for BP prediction.
    5. Tokenizes the sequences using the provided tokenizer.
    6. Constructs dataset objects for model training.

    Args:
        df (pd.DataFrame): input dataframe containing the intron data.
        tokenizer: tokenizer used to convert sequences into model-compatible input.
        model: the model to be trained.
        max_model_input_size (int): maximum token length allowed for the model input.
        seed (int): random seed for reproducibility when sampling or shuffling.
        sample_n_datapoints (int, optional): number of datapoints to randomly sample. Defaults to None, which represents the entire dataset.
        truncate (bool, optional): whether to truncate sequences to fit `max_model_input_size`. Defaults to True.
        shuffle (bool, optional): whether to shuffle data splits. Defaults to True.

    Returns:
        tuple: three HuggingFace Dataset objects corresponding to the train, validation, and test sets that will be used to train the model.

    Example:
        train_dataset, val_dataset, test_dataset = create_dataset(
            df=dataframe, 
            tokenizer=my_tokenizer, 
            model=my_model, 
            max_model_input_size=512, 
            seed=42
        )
    """

    if sample_n_datapoints:
        df = df.sample(n = sample_n_datapoints, random_state=seed)
    
    # Calculate BP_POS_WITHIN_STRAND
    df['BP_POS_WITHIN_STRAND'] = df['IVS_SIZE'] + df['BP_ACC_DIST']

    # Remove all data points where the BP is farther than
    df = df[df['IVS_SIZE'] - df['BP_POS_WITHIN_STRAND'] <= max_model_input_size]

    # Split to train/val/test
    train_df, val_df, test_df = split_train_test_on_chr(df, shuffle, seed)

    # Extract the sequences and labels that the model will train on
    train_seqs, train_labels = extract_intron_seq_and_labels(train_df, max_model_input_size, truncate=True)
    test_seqs, test_labels = extract_intron_seq_and_labels(test_df, max_model_input_size, truncate=True)
    val_seqs, val_labels = extract_intron_seq_and_labels(val_df, max_model_input_size, truncate=True)

    # Tokenize the sequences. For splicebert testing use tokenizer(train_seqs, padding_side = 'left')
    train_ids = tokenizer(train_seqs, padding = 'max_length', truncation = True, padding_side = 'left', max_length = max_model_input_size, return_tensors = 'pt')
    test_ids = tokenizer(test_seqs, padding = 'max_length', truncation = True, padding_side = 'left', max_length = max_model_input_size, return_tensors = 'pt')
    val_ids = tokenizer(val_seqs, padding = 'max_length', truncation = True, padding_side = 'left', max_length = max_model_input_size, return_tensors = 'pt')

    # Build the dataset structure that will be passed for training
    train_dataset = Dataset.from_dict(train_ids)
    train_dataset = train_dataset.add_column("labels", train_labels)

    val_dataset = Dataset.from_dict(val_ids)
    val_dataset = val_dataset.add_column("labels", val_labels)

    test_dataset = Dataset.from_dict(test_ids)
    test_dataset = test_dataset.add_column("labels", test_labels)

    return train_dataset, val_dataset, test_dataset
