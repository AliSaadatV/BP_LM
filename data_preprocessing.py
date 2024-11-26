# Helper file to do some data preprocessing and preperation for training

def preprocess_data(df):
    """
    Basic preprocessing of data. We remove "BP_POS" and "BP_ACC_SEQ" as there are
    no longer accurate due to "flipping" the negative strands.

    NOTE: The added column BP_POS_WITHIN_STRAND is ZERO-INDEXED

    Args:
        df: a dataframe of the data
    """
    # Replace all the strand entries with 0 (negative) or 1 (positive)
    df['STRAND'] = df['STRAND'].replace({'+': 1, '-': 0})

    # For negative strands, the BP is on the left (see "More Data Analysis..." in dataexploration.ipynb")

    # Flip DNA sequences for rows where Strand is '0'
    df['IVS_SEQ'] = df.apply(
        lambda row: row['IVS_SEQ'][::-1] if row['STRAND'] == 0 else row['IVS_SEQ'],
        axis=1
    )

    # Add a column for the BP position within the intron strand (IVS_SEQ), 0-indexed
    # Note, the calculation is different depending on whether the strand is pos or neg
    def compute_BP_pos_within_strand(row):
        if row['STRAND'] == 1:
            return row['BP_POS'] - row['START']
        else:
            return row['END'] - row['BP_POS']
    
    df['BP_POS_WITHIN_STRAND'] = df.apply(compute_BP_pos_within_strand, axis=1)

    # Delete these columns as they are no longer accurate for negative strands
    columns_to_delete = ['BP_POS', 'BP_ACC_SEQ']
    df.drop(columns=columns_to_delete, inplace=True)


def split_train_test_on_chr(df, train_chrs, val_chrs, test_chrs, shuffle=True):

    """
    Creates a train/val/test split specified by the chromosome types.
    Note: Should run 'extract_intron_seq_and_labels on each dataframe after this method.

    Args:
        df: a dataframe of the data
        train_chr [str]: List of chromosomes to use for training
        val_chr [str]: List of chromosomes to use for validation
        test_chr [str]: List of chromosomes to use for testing

    Returns:
        train_df:
        val_df:
        test_df:

    """
    # Check that there is no overlap in chromosomes between each set
    assert not (set(train_chrs) & set(val_chrs)), f"Overlap found between train and val sets: {set(train_chrs) & set(val_chrs)}"
    assert not (set(train_chrs) & set(test_chrs)), f"Overlap found between train and test sets: {set(train_chrs) & set(test_chrs)}"
    assert not (set(val_chrs) & set(test_chrs)), f"Overlap found between val and test sets: {set(val_chrs) & set(test_chrs)}"

    # Split the dataframe based on chromosomes
    train_df = df[df["CHR"].isin(train_chrs)]
    val_df = df[df["CHR"].isin(val_chrs)]
    test_df = df[df["CHR"].isin(test_chrs)]

    if shuffle:
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        val_df = val_df.sample(frac=1).reset_index(drop=True)
        test_df = test_df.sample(frac=1).reset_index(drop=True)

    # Output statistics on the dataset
    print("Chromosomes in train set:", set(train_df["CHR"]))
    print("Chromosomes in validation set:", set(val_df["CHR"]))
    print("Chromosomes in test set:", set(test_df["CHR"]))

    train_points = len(train_df)
    val_points = len(val_df)
    test_points = len(test_df)
    total_points = train_points + val_points + test_points

    print("\nTotal data points:", train_points + val_points + test_points)
    print(f"Train set contains {train_points} data points ({(train_points / total_points) * 100:.2f}%)")
    print(f"Validation set contains {val_points} data points ({(val_points / total_points) * 100:.2f}%)")
    print(f"Test set contains {test_points} data points ({(test_points / total_points) * 100:.2f}%)")

    return train_df, val_df, test_df


def extract_intron_seq_and_labels(df, max_model_input_size=0, truncate=True):
    """
    Extract the intron sequence (IVS_SEQ) and the BP location (BP_WITHIN_STRAND).
    This should be run after splitting train/val/test, as the gene and chromosome 
    information is removed. Also truncates intron strands and its corresponding labels

    Args:
        df: a dataframe of the data (either train, val, or test)
        max_model_input_size: the maximum token size the model can take
        truncate: whether to truncate introns/labels

    Returns:
        ivs_seq_list: list of the intron sequences
        labels: list of the BP pos, where each i'th entry is a list of all zeros except for a 1 at the BP
    """

    # Select the relevant columns
    ivs_seq_list = df['IVS_SEQ'].tolist()
    bp_pos_within_strand_list = df['BP_POS_WITHIN_STRAND'].tolist()

    #Create binary label for each IVS_SEQ
    #labels = [
    #    [].append([1 if i == bp_pos else 0 for i in range(len(seq))])
    #    for seq, bp_pos in zip(ivs_seq_list, bp_pos_within_strand_list)
    #]
    
    labels = [bp_pos*[0] + [1] + (len(seq)-bp_pos-1)*[0]
              for seq, bp_pos in zip(ivs_seq_list, bp_pos_within_strand_list)
    ]

    if truncate:
        ivs_seq_list, labels = truncate_strands(
            ivs_seq_list, 
            labels, 
            max_model_input_size)

    # Result as two lists
    return ivs_seq_list, labels


def truncate_strands(intron_list, labels, max_length):
    """
    Truncate intron sequence and corresponding labels from the right such that
    the intron sequence fits within the model. We are truncating from the right
    since all branch points are located on the right side.

    Args:
        intron_list:
        labels:
        max_length: max token length of model (model must have token sizes of 1 nucleotide)
    """
    truncated_introns = []
    truncated_labels = []

    for intron, label in zip(intron_list, labels):
        if(len(intron) > max_length):
            truncated_introns.append(intron[(len(intron) - max_length):])
            truncated_labels.append(label[(len(intron) - max_length):])

        else:
            truncated_introns.append(intron)
            truncated_labels.append(label)

    return truncated_introns, truncated_labels
