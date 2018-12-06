import numpy as np

import dfutils as dfu

def prepare_datas(df, holdout_df=None, features=None, feature_encoding=None,
        feature_standard_scaling=None,
        label_col=None, one_hot_encoding=None):
    '''
ipdb> pp classifier.fit(np.array(datas['X_train']), 
                    np.array(datas['y_train'][u'end station id']))

preparing a holdout set:
    - reuse the label encoders from the training, but for the holdout set,
    need to check if values are not in the encoding set, then need to replace w/ -1 .
    '''
    # Remove nulls...
    df_unnulled = dfu.remove_rows_with_nulls(df)
    df_re_index = dfu.re_index(df_unnulled)
    df = df_re_index

    # Remove nulls from holdout too
    if holdout_df is not None:
        holdout_df_unnulled = dfu.remove_rows_with_nulls(holdout_df)
        holdout_df_re_index = dfu.re_index(holdout_df_unnulled)
        holdout_df = holdout_df_re_index

    if feature_encoding:
        label_encoders = {}

        # Get the encoed dataframe.
        dfcopy, label_encoders = build_label_encoders_from_df(df, feature_encoding)
        df = dfcopy

        if holdout_df is not None:
            holdout_df = encode_holdout_df(holdout_df, label_encoders, feature_encoding)

    if features:
        X = df[features]

        if holdout_df is not None:
            X_holdout = holdout_df[features]
    else:
        X = df[df.columns[:-1]]

        if holdout_df is not None:
            X_holdout = holdout_df[holdout_df.columns[:-1]]

    # One hot here.
    if one_hot_encoding:
        # For each input feature being encoded, we want the new columns concatenated
        #   into the output.
        oh_encoders = pl.make_one_hot_encoders(X, one_hot_encoding)
        X = pl.feature_binarization(X, oh_encoders)

        if X_holdout is not None:
            # X_holdout = pl.feature_binarization(X_holdout, one_hot_encoding)
            X_holdout = pl.feature_binarization(X_holdout, oh_encoders)

    # FIXME ... use the appropriate LabelEncoder here. Unless already done by the t_t_split()
    if label_col:
        y = df[label_col]

        if holdout_df is not None:
            y_holdout = holdout_df[label_col]
    else:
        y = df[df.columns[-1]]

        if holdout_df is not None:
            y_holdout = holdout_df[holdout_df.columns[-1]]

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)

    if feature_standard_scaling:
        # Train 
        scaler = StandardScaler().fit(X_train)
        X_train_out  = scaler.transform(X_train)
        X_test_out = scaler.transform(X_test)

        # TODO: same scaler or different scaler on whole X?
        #   probably on whole X, and then that scaler needs to be
        #   stored away for transforming new input vectors which come in to be predicted.
        scaler_full = StandardScaler().fit(X)
        X_full = scaler_full.transform(X)

        # 
        if holdout_df is not None:
            X_holdout_scaled = scaler_full.transform(X_holdout)
    else:
        X_train_out = X_train
        X_test_out = X_test
        X_full = X

        if holdout_df is not None:
            X_holdout_scaled = X_holdout
       
    dump_np_array(X_full, 'dump')

    # TODO: how to also return the label encoders and scalers

    out = {
            # All data
            'X': np.array(X_full),
            'y': np.array(y),
            'X_train': np.array(X_train_out), 
            'X_test': np.array(X_test_out), 
            'y_train': np.array(y_train), 
            'y_test': np.array(y_test),
            }

    if holdout_df is not None:
        out.update({
            'X_holdout': np.array(X_holdout_scaled),
            'y_holdout': np.array(y_holdout)
            })

    return out

