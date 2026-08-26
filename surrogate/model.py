"""Small NumPy-only inference adapter for LegacyPlume surrogate archives.

The archive format is intentionally explicit and pickle-free.  A trained model
must contain feature_names, target_names, x_mean, x_scale, y_mean and y_scale.
It may then contain either a linear map (coef, intercept), or a dense MLP whose
layers are W0/b0, W1/b1, ... .  Hidden MLP layers use tanh; the last layer is
linear.
"""

from __future__ import print_function

import json

import numpy as np


FORMAT_NAME = "legacyplume-surrogate-v1"


def _string_list(value, label):
    arr = np.asarray(value)
    if arr.ndim == 0:
        raw = arr.item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            parsed = json.loads(str(raw))
        except ValueError:
            parsed = [str(raw)]
        return [str(item) for item in parsed]
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in arr.tolist()]


class NpzSurrogate(object):
    """Validated, vectorized predictor backed by a ``numpy.savez`` archive."""

    def __init__(self, path):
        self.path = str(path)
        with np.load(self.path, allow_pickle=False) as data:
            self.arrays = {key: np.asarray(data[key]) for key in data.files}

        required = ("feature_names", "target_names", "x_mean", "x_scale",
                    "y_mean", "y_scale")
        missing = [key for key in required if key not in self.arrays]
        if missing:
            raise ValueError(
                "Surrogate archive is missing: %s. Expected %s."
                % (", ".join(missing), FORMAT_NAME)
            )

        self.feature_names = _string_list(
            self.arrays["feature_names"], "feature_names")
        self.target_names = _string_list(
            self.arrays["target_names"], "target_names")
        self.x_mean = self._vector("x_mean", len(self.feature_names))
        self.x_scale = self._vector("x_scale", len(self.feature_names))
        self.y_mean = self._vector("y_mean", len(self.target_names))
        self.y_scale = self._vector("y_scale", len(self.target_names))

        if np.any(self.x_scale == 0.0) or np.any(self.y_scale == 0.0):
            raise ValueError("x_scale and y_scale must not contain zero")

        has_linear = "coef" in self.arrays
        has_mlp = "W0" in self.arrays
        if has_linear == has_mlp:
            raise ValueError(
                "Archive must contain exactly one model form: coef/intercept "
                "or consecutive W0/b0, W1/b1, ... layers"
            )
        self.kind = "linear" if has_linear else "mlp"
        self._validate_model()

    def _vector(self, key, size):
        value = np.asarray(self.arrays[key], dtype=float).reshape(-1)
        if value.size != size:
            raise ValueError("%s has %d values; expected %d" %
                             (key, value.size, size))
        return value

    def _validate_model(self):
        n_in = len(self.feature_names)
        n_out = len(self.target_names)
        if self.kind == "linear":
            coef = np.asarray(self.arrays["coef"], dtype=float)
            if coef.shape == (n_out, n_in):
                coef = coef.T
            if coef.shape != (n_in, n_out):
                raise ValueError("coef must have shape (%d, %d)" %
                                 (n_in, n_out))
            self.coef = coef
            self.intercept = self._vector("intercept", n_out)
            return

        self.layers = []
        width = n_in
        index = 0
        while "W%d" % index in self.arrays:
            w_key, b_key = "W%d" % index, "b%d" % index
            if b_key not in self.arrays:
                raise ValueError("Missing %s" % b_key)
            weights = np.asarray(self.arrays[w_key], dtype=float)
            bias = np.asarray(self.arrays[b_key], dtype=float).reshape(-1)
            if weights.ndim != 2 or weights.shape[0] != width:
                raise ValueError("%s has incompatible shape %s" %
                                 (w_key, weights.shape))
            if weights.shape[1] != bias.size:
                raise ValueError("%s and %s widths differ" % (w_key, b_key))
            self.layers.append((weights, bias))
            width = bias.size
            index += 1
        layer_keys = sorted(key for key in self.arrays if
                            key.startswith("W") and key[1:].isdigit())
        if set(layer_keys) != set("W%d" % i for i in range(index)):
            raise ValueError("MLP layer numbering must be consecutive")
        if not self.layers or width != n_out:
            raise ValueError("Final MLP width must equal target count %d" % n_out)

    def predict(self, features):
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise ValueError("features must have shape (n, %d)" %
                             len(self.feature_names))
        z = (x - self.x_mean) / self.x_scale
        if self.kind == "linear":
            y = np.dot(z, self.coef) + self.intercept
        else:
            y = z
            for index, (weights, bias) in enumerate(self.layers):
                y = np.dot(y, weights) + bias
                if index + 1 != len(self.layers):
                    y = np.tanh(y)
        return y * self.y_scale + self.y_mean

    def target_index(self, *aliases):
        lowered = [name.lower() for name in self.target_names]
        for alias in aliases:
            if alias.lower() in lowered:
                return lowered.index(alias.lower())
        return None
