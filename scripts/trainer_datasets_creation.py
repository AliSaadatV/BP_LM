from datasets import Dataset
from .data_preprocessing import *

def create_dataset(df, tokenizer, model, max_model_input_size, seed, sample_n_datapoints=None, truncate=True, shuffle=True):
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

    # Tokenize the sequences
    train_ids = tokenizer(train_seqs, padding_side = 'left')
    test_ids = tokenizer(test_seqs, padding_side = 'left')
    val_ids = tokenizer(val_seqs, padding_side = 'left')

    # Build the dataset structure that will be passed for training
    train_dataset = Dataset.from_dict(train_ids)
    train_dataset = train_dataset.add_column("labels", train_labels)

    val_dataset = Dataset.from_dict(val_ids)
    val_dataset = val_dataset.add_column("labels", val_labels)

    test_dataset = Dataset.from_dict(test_ids)
    test_dataset = test_dataset.add_column("labels", test_labels)

    return train_dataset, val_dataset, test_dataset