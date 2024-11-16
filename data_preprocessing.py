# Helper file to do some data preprocessing

def preprocess_data(df):
    """
    Args:
        df: a dataframe of the data
    """

    # Replace all the strand entries with 0 (negative) or 1 (positive)
    df['STRAND'] = df['STRAND'].replace({'+': 1, '-': 0})

    # Add a column for the BP position within the intron strand (IVS_SEQ), 0-indexed
    df['BP_POS_WITHIN_STRAND'] = df['BP_POS'] - df['START']

    # Not sure how else to preprocess data, since it depends on the pretrained models
    # we use and the data format that it was trained on (TODO: explore this?)