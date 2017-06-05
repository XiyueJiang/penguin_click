

nn = all_data.shape[0]
np.random.seed(999)
sample_idx = np.random.random_integers(0, 3, nn)
n_trees = 4100
predv_xgb = 0
batch = 0
day_test = 31

output_logloss = {}
pred_dict = {}

for idx in [0, 1, 2, 3]:
    filter1 = np.logical_and(np.logical_and(day_values >= 17, day_values < day_test),
                             np.logical_and(sample_idx == idx, True))
    filter_v1 = day_values == day_test
    xt1 = all_data.ix[filter1, xgb_feature]
    yt1 = cvrt_value[filter1]

    xv1 = all_data.ix[filter_v1, xgb_feature]
    yv1 = cvrt_value[filter_v1]

    if xt1.shape[0] <= 0 or xt1.shape[0] != yt1.shape[0]:
        print(xt1.shape, yt1.shape)
        raise ValueError('wrong shape!')

    dtrain = xgb.DMatrix(xt1, label=yt1)
    dvalid = xgb.DMatrix(xv1)
    watchlist = [(dtrain, 'train')]
    print(xt1.shape, yt1.shape)
    plst = list(xgb_param.items()) + [('eval_metric', 'logloss')]
    xgb1 = xgb.train(plst, dtrain, n_trees, watchlist, early_stopping_rounds=50)
    batch += 1
    current_pred = xgb1.predict(dvalid)
    yt_hat = xgb1.predict(dtrain)

    pred_dict[idx] = current_pred
    predv_xgb += current_pred

    output_logloss[idx] = logloss(yt_hat, yt1)

    print(logloss(yt_hat, yt1))

    # print('-' * 30, batch, logloss(predv_xgb / batch, yv1))


