'xgb-tst': {
        'features': ['numeric'],
        'model': Xgb({'max_depth': 5, 'eta': 0.05}, n_iter=10),
        'param_grid': {'colsample_bytree': [0.2, 1.0]},
    },

    'xgb2': {
        'features': ['numeric', 'categorical_counts'],
        'y_transform': y_norm,
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.1,
            'colsample_bytree': 0.5,
            'subsample': 0.95,
            'min_child_weight': 5,
        }, n_iter=400),
        'param_grid': {'colsample_bytree': [0.2, 1.0]},
    },

    'xgb-ce': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_norm,
        'n_bags': 2,
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.03,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 2,
            'gamma': 0.2,
        }, n_iter=2000),
    },

    'xgb-ce-2': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log_ofs(200),
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 12,
            'eta': 0.01,
            'colsample_bytree': 0.5,
            'subsample': 0.8,
            'gamma': 1,
            'alpha': 1,
        }, n_iter=3000),
    },

    'xgb-ce-3': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log_ofs(200),
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 14,
            'eta': 0.01,
            'colsample_bytree': 0.5,
            'subsample': 0.8,
            'gamma': 1.5,
            'alpha': 1,
        }, n_iter=3000),
    },

    'xgb-ce-tst': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log_ofs(200),
        'n_bags': 2,
        'model': Xgb({
            'max_depth': 14,
            'eta': 0.01,
            'colsample_bytree': 0.5,
            'subsample': 0.8,
            'gamma': 1.5,
            'alpha': 1,
        }, n_iter=3000),
    },

    'xgb4': {
        'features': ['numeric', 'categorical_dummy'],
        'y_transform': y_norm,
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.02,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 2,
        }, n_iter=3000),
    },

    'xgb6': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_norm,
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.03,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 4,
        }, n_iter=2000),
        'param_grid': {'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]},
    },

    'xgb7': {
        'features': ['numeric'],
        'feature_builders': [CategoricalMeanEncoded(C=10000, noisy=False, loo=False)],
        'y_transform': y_norm,
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.03,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 4,
        }, n_iter=2000),
        'param_grid': {'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]},
    },

    'xgbh-ce': {
        'features': ['numeric', 'categorical_encoded'],
        #'n_bags': 2,
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.05,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 4,
        }, n_iter=2000, huber=100),
        'param_grid': {'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]},
    },

    'xgbf-ce': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_norm,
        #'n_bags': 3,
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.05,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 4,
            'alpha': 0.0005,
        }, n_iter=1100, fair=1),
        'param_grid': {'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]},
    },

    'xgbf-ce-2': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_norm,
        'n_bags': 2,
        'model': Xgb({
            'max_depth': 8,
            'eta': 0.04,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'gamma': 0.45,
            'alpha': 0.0005,
        }, n_iter=1320, fair=1),
        'param_grid': {'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]},
    },

    'xgbf-ce-3': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_norm,
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 12,
            'eta': 0.02,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'gamma': 0.5,
            'alpha': 0.5,
        }, n_iter=4000, fair=1),
        'param_grid': {'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]},
    },

    'xgbf-ce-4': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_norm,
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 15,
            'eta': 0.02,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'gamma': 0.6,
            'alpha': 0.5,
        }, n_iter=5400, fair=1),
        'param_grid': {'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]},
    },

    'xgbf-ce-4-2': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log,
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 15,
            'eta': 0.02,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'gamma': 1.5,
            'alpha': 1.0,
        }, n_iter=5400, fair=1),
        'param_grid': {'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]},
    },

    'xgbf-ce-5': {
        'features': ['numeric', 'categorical_encoded'],
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 9,
            'eta': 0.01,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'alpha': 0.9,
        }, n_iter=6500, fair=150, fair_decay=0.0003),
    },

    'xgbf-ce-6': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13')]
            )],
        'y_transform': y_norm,
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 13,
            'eta': 0.02,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'gamma': 0.6,
            'alpha': 0.5,
        }, n_iter=5400, fair=1),
    },

    'xgbf-ce-7': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13')]
            )],
        'y_transform': y_log_ofs(200),
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 13,
            'eta': 0.02,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'gamma': 1.2,
            'alpha': 1.0,
        }, n_iter=5000, fair=1),
    },

    'xgbf-ce-8': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13')]
            )],
        'y_transform': y_log_ofs(200),
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 13,
            'eta': 0.01,
            'colsample_bytree': 0.2,
            'subsample': 0.95,
            'gamma': 1.15,
            'alpha': 1.0,
        }, n_iter=16000, fair=1),
    },

    'xgbf-ce-9': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13'), ('cat79', 'cat81'), ('cat81', 'cat13'), ('cat9', 'cat73'), ('cat2', 'cat81'), ('cat80', 'cat111'), ('cat79', 'cat111'), ('cat72', 'cat1'), ('cat23', 'cat103'), ('cat89', 'cat13'), ('cat57', 'cat14'), ('cat80', 'cat81'), ('cat81', 'cat11'), ('cat9', 'cat103'), ('cat23', 'cat36')]
            )],
        'y_transform': y_log_ofs(200),
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 12,
            'eta': 0.01,
            'colsample_bytree': 0.2,
            'subsample': 0.95,
            'gamma': 1.1,
            'alpha': 0.95,
        }, n_iter=16000, fair=1),
    },

    'xgbf-ce-10': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=itertools.combinations('cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,cat81,cat11,cat1,cat13,cat9,cat3,cat16,cat90,cat23,cat36,cat73,cat103,cat40,cat28,cat111,cat6,cat76,cat50,cat5,cat4,cat14,cat38,cat24,cat82,cat25'.split(','), 2)
            )],
        'y_transform': y_log_ofs(200),
        'n_bags': 6,
        'model': Xgb({
            'max_depth': 12,
            'eta': 0.03,
            'colsample_bytree': 0.7,
            'subsample': 0.7,
            'min_child_weight': 100
        }, n_iter=720, fair=0.7),
    },

    'xgbf-ce-11': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13'), ('cat79', 'cat81'), ('cat81', 'cat13'), ('cat9', 'cat73'), ('cat2', 'cat81'), ('cat80', 'cat111'), ('cat79', 'cat111'), ('cat72', 'cat1'), ('cat23', 'cat103'), ('cat89', 'cat13'), ('cat57', 'cat14'), ('cat80', 'cat81'), ('cat81', 'cat11'), ('cat9', 'cat103'), ('cat23', 'cat36')]
            )],
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 12,
            'eta': 0.04,
            'colsample_bytree': 0.2,
            'subsample': 0.75,
            'gamma': 2.0,
            'alpha': 2.0,
        }, n_iter=10000, fair=200),
    },

    'xgbf-ce-12': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13')]
            )],
        'y_transform': y_norm,
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 13,
            'eta': 0.01,
            'colsample_bytree': 0.2,
            'subsample': 0.95,
            'gamma': 1.15,
            'alpha': 1.0,
        }, n_iter=16000, fair=1),
    },

    'xgbf-ce-13': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13'), ('cat79', 'cat81'), ('cat81', 'cat13'), ('cat9', 'cat73'), ('cat2', 'cat81'), ('cat80', 'cat111'), ('cat79', 'cat111'), ('cat72', 'cat1'), ('cat23', 'cat103'), ('cat89', 'cat13'), ('cat57', 'cat14'), ('cat80', 'cat81'), ('cat81', 'cat11'), ('cat9', 'cat103'), ('cat23', 'cat36')]
            )],
        'y_transform': y_pow(0.25),
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 12,
            'eta': 0.007,
            'colsample_bytree': 0.2,
            'subsample': 0.95,
            'gamma': 2.2,
            'alpha': 1.2,
        }, n_iter=8000, fair=1),
    },

    'xgbf-ce-14': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat101', 'cat80'), ('cat79', 'cat80'), ('cat12', 'cat80'), ('cat101', 'cat81'), ('cat12', 'cat81'), ('cat12', 'cat79'), ('cat57', 'cat79'), ('cat1', 'cat80'), ('cat101', 'cat79'), ('cat1', 'cat81')]
            )],
        'y_transform': y_pow_ofs(0.202, 5),
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 12,
            'eta': 0.01,
            'colsample_bytree': 0.2,
            'subsample': 0.95,
            'gamma': 1.3,
            'alpha': 0.6,
        }, n_iter=8000, fair=2.0),
    },

    'xgbf-ce-15': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13'), ('cat79', 'cat81'), ('cat81', 'cat13'), ('cat9', 'cat73'), ('cat2', 'cat81'), ('cat80', 'cat111'), ('cat79', 'cat111'), ('cat72', 'cat1'), ('cat23', 'cat103'), ('cat89', 'cat13'), ('cat57', 'cat14'), ('cat80', 'cat81'), ('cat81', 'cat11'), ('cat9', 'cat103'), ('cat23', 'cat36')]
            )],
        'y_transform': y_pow(0.24),
        'n_bags': 6,
        'model': Xgb({
            'max_depth': 12,
            'eta': 0.005,
            'colsample_bytree': 0.2,
            'subsample': 0.95,
            'gamma': 2.2,
            'alpha': 1.2,
        }, n_iter=11000, fair=1),
    },
