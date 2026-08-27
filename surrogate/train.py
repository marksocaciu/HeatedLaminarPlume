#!/usr/bin/env python
"""Train a NumPy ridge surrogate and export LegacyPlume's model archive.

This is a deliberately small baseline trainer.  It accepts either two matrices
from a dataset archive, or lists of individual one-dimensional array keys.  The
exported archive is consumed directly by ``surrogate.generate_initial_state``.
"""

from __future__ import print_function

import argparse
import json

import numpy as np


def _names(value):
    array = np.asarray(value)
    if array.ndim == 0:
        raw = array.item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            return [str(item) for item in json.loads(str(raw))]
        except ValueError:
            return [part.strip() for part in str(raw).split(",") if part.strip()]
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in array.tolist()]


def _csv_names(text):
    if not text:
        return None
    return [part.strip() for part in text.split(",") if part.strip()]


def _matrix(data, matrix_key, column_keys, label):
    if matrix_key:
        if matrix_key not in data:
            raise KeyError("Dataset has no %s key %r; available keys: %s" %
                           (label, matrix_key, ", ".join(data.files)))
        result = np.asarray(data[matrix_key], dtype=float)
        if result.ndim == 1:
            result = result.reshape((-1, 1))
        if result.ndim != 2:
            raise ValueError("%s matrix %r must be two-dimensional" %
                             (label, matrix_key))
        return result

    keys = _csv_names(column_keys)
    if not keys:
        raise ValueError("Specify --%s-key or --%s-columns" % (label, label))
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError("Missing %s columns: %s" % (label, ", ".join(missing)))
    columns = [np.asarray(data[key], dtype=float).reshape(-1) for key in keys]
    sizes = set(column.size for column in columns)
    if len(sizes) != 1:
        raise ValueError("All %s columns must have the same length" % label)
    return np.column_stack(columns)


def _resolve_names(data, explicit, archive_key, column_keys, width, label):
    names = _csv_names(explicit)
    if names is None and archive_key in data:
        names = _names(data[archive_key])
    if names is None and column_keys:
        names = _csv_names(column_keys)
    if names is None:
        raise ValueError(
            "No %s names found. Supply --%s-names in model input order." %
            (label, label)
        )
    if len(names) != width:
        raise ValueError("Got %d %s names for matrix width %d" %
                         (len(names), label, width))
    return names


def _safe_scale(values):
    scale = np.std(values, axis=0)
    scale[scale < 1.0e-14] = 1.0
    return scale


def fit_ridge(x_train, y_train, ridge):
    """Fit normalized multi-output ridge regression with an intercept."""
    x_mean = np.mean(x_train, axis=0)
    x_scale = _safe_scale(x_train)
    y_mean = np.mean(y_train, axis=0)
    y_scale = _safe_scale(y_train)
    x_norm = (x_train - x_mean) / x_scale
    y_norm = (y_train - y_mean) / y_scale

    design = np.column_stack((x_norm, np.ones(x_norm.shape[0])))
    gram = np.dot(design.T, design)
    penalty = float(ridge) * np.eye(gram.shape[0])
    penalty[-1, -1] = 0.0
    solution = np.linalg.solve(
        gram + penalty, np.dot(design.T, y_norm))
    return {
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "coef": solution[:-1, :],
        "intercept": solution[-1, :],
    }


def predict(x, fitted):
    normalized = (x - fitted["x_mean"]) / fitted["x_scale"]
    y_norm = np.dot(normalized, fitted["coef"]) + fitted["intercept"]
    return y_norm * fitted["y_scale"] + fitted["y_mean"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--features-key",
                    help="Dataset key containing the complete feature matrix")
    ap.add_argument("--targets-key",
                    help="Dataset key containing the complete target matrix")
    ap.add_argument("--features-columns",
                    help="Comma-separated dataset keys to stack as features")
    ap.add_argument("--targets-columns",
                    help="Comma-separated dataset keys to stack as targets")
    ap.add_argument("--feature-names",
                    help="Comma-separated model feature names, in matrix order")
    ap.add_argument("--target-names",
                    help="Comma-separated model target names, in matrix order")
    ap.add_argument("--ridge", type=float, default=1.0e-8)
    ap.add_argument("--validation-fraction", type=float, default=0.1)
    ap.add_argument("--holdout-key",
                    help="Optional per-row dataset key used for a complete holdout")
    ap.add_argument("--holdout-value",
                    help="Value in --holdout-key to reserve for validation")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    if args.ridge < 0.0:
        raise ValueError("--ridge must be nonnegative")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must lie between zero and one")

    with np.load(args.dataset, allow_pickle=False) as data:
        x = _matrix(data, args.features_key, args.features_columns, "features")
        y = _matrix(data, args.targets_key, args.targets_columns, "targets")
        feature_names = _resolve_names(
            data, args.feature_names, "feature_names", args.features_columns,
            x.shape[1], "feature")
        target_names = _resolve_names(
            data, args.target_names, "target_names", args.targets_columns,
            y.shape[1], "target")
        holdout = None
        if args.holdout_key:
            if args.holdout_value is None:
                raise ValueError("--holdout-key requires --holdout-value")
            if args.holdout_key not in data:
                raise KeyError("Dataset has no holdout key %r" % args.holdout_key)
            holdout = np.asarray(data[args.holdout_key]).reshape(-1)
        elif args.holdout_value is not None:
            raise ValueError("--holdout-value requires --holdout-key")

    if x.shape[0] != y.shape[0]:
        raise ValueError("Feature and target matrices have different row counts")
    finite = np.all(np.isfinite(x), axis=1) & np.all(np.isfinite(y), axis=1)
    if holdout is not None and holdout.size != x.shape[0]:
        raise ValueError("Holdout array and feature matrix have different row counts")
    removed = int(x.shape[0] - np.count_nonzero(finite))
    x, y = x[finite], y[finite]
    if holdout is not None:
        holdout = holdout[finite]
    if x.shape[0] < 3:
        raise ValueError("Too few finite samples for training")

    if holdout is not None:
        # Compare as strings so numeric and string-valued experiment labels both work.
        validation_mask = np.asarray([str(value) == str(args.holdout_value)
                                      for value in holdout], dtype=bool)
        validation_rows = np.nonzero(validation_mask)[0]
        training_rows = np.nonzero(~validation_mask)[0]
        if validation_rows.size == 0:
            raise ValueError("No rows match holdout value %r" % args.holdout_value)
        if training_rows.size == 0:
            raise ValueError("Holdout selection leaves no training rows")
    else:
        rng = np.random.RandomState(args.seed)
        order = rng.permutation(x.shape[0])
        n_validation = max(1, int(round(args.validation_fraction * x.shape[0])))
        validation_rows = order[:n_validation]
        training_rows = order[n_validation:]
    fitted = fit_ridge(x[training_rows], y[training_rows], args.ridge)
    validation_prediction = predict(x[validation_rows], fitted)
    residual = validation_prediction - y[validation_rows]
    rmse = np.sqrt(np.mean(residual ** 2, axis=0))
    target_std = _safe_scale(y[validation_rows])
    normalized_rmse = rmse / target_std

    np.savez(
        args.output,
        format_name=np.array("legacyplume-surrogate-v1"),
        feature_names=np.asarray(feature_names),
        target_names=np.asarray(target_names),
        x_mean=fitted["x_mean"],
        x_scale=fitted["x_scale"],
        y_mean=fitted["y_mean"],
        y_scale=fitted["y_scale"],
        coef=fitted["coef"],
        intercept=fitted["intercept"],
        training_samples=np.array(training_rows.size),
        validation_samples=np.array(validation_rows.size),
        validation_rmse=rmse,
        validation_normalized_rmse=normalized_rmse,
        ridge=np.array(args.ridge),
        random_seed=np.array(args.seed),
        holdout_key=np.array(args.holdout_key or ""),
        holdout_value=np.array(args.holdout_value or ""),
    )

    print("Removed non-finite samples: %d" % removed)
    print("Training samples: %d" % training_rows.size)
    print("Validation samples: %d" % validation_rows.size)
    for name, value, normalized in zip(target_names, rmse, normalized_rmse):
        print("  %s: RMSE=%.6e, normalized_RMSE=%.6e" %
              (name, value, normalized))
    print("Wrote trained surrogate: %s" % args.output)


if __name__ == "__main__":
    main()
