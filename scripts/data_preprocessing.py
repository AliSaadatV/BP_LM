# Helper file to do some data preprocessing and preperation for training
from datasets import Dataset

# Create a split based on chromosome types
TRAIN_CHRS = ["chr1", "chr2", "chr3", "chr4",
              "chr5", "chr6", "chr7", "chr12", 
              "chr13", "chr14", "chr15", "chr16", 
              "chr17", "chr18", "chr19", "chr20", 
              "chr21", "chr22", "chrX", "chrY"]

# This gives a 80/10/10 train/val/test split. Add chr6 and chr7 for a 70/15/15 split
VAL_CHRS = ["chr9", "chr10"]
TEST_CHRS = ["chr8", "chr11"]

def split_train_test_on_chr(df, shuffle=False, seed=None):
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
    assert not (set(TRAIN_CHRS) & set(VAL_CHRS)), f"Overlap found between train and val sets: {set(train_chrs) & set(val_chrs)}"
    assert not (set(TRAIN_CHRS) & set(TEST_CHRS)), f"Overlap found between train and test sets: {set(train_chrs) & set(test_chrs)}"
    assert not (set(VAL_CHRS) & set(TEST_CHRS)), f"Overlap found between val and test sets: {set(val_chrs) & set(test_chrs)}"

    # Split the dataframe based on chromosomes
    train_df = df[df["CHR"].isin(TRAIN_CHRS)]
    val_df = df[df["CHR"].isin(VAL_CHRS)]
    test_df = df[df["CHR"].isin(TEST_CHRS)]

    if shuffle:
        train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        val_df = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)

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
