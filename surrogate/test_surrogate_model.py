import numpy as np

from surrogate.model import NpzSurrogate


def test_linear_archive_round_trip(tmp_path):
    path = tmp_path / "linear.npz"
    np.savez(
        str(path),
        feature_names=np.array(["x_star", "Pr"]),
        target_names=np.array(["u_star", "theta_star"]),
        x_mean=np.array([1.0, 2.0]),
        x_scale=np.array([2.0, 4.0]),
        y_mean=np.array([10.0, 20.0]),
        y_scale=np.array([2.0, 3.0]),
        coef=np.eye(2),
        intercept=np.zeros(2),
    )
    model = NpzSurrogate(str(path))
    result = model.predict(np.array([[3.0, 6.0]]))
    np.testing.assert_allclose(result, [[12.0, 23.0]])


def test_tanh_mlp_archive(tmp_path):
    path = tmp_path / "mlp.npz"
    np.savez(
        str(path),
        feature_names=np.array(["x_star"]),
        target_names=np.array(["theta_star"]),
        x_mean=np.zeros(1), x_scale=np.ones(1),
        y_mean=np.zeros(1), y_scale=np.ones(1),
        W0=np.array([[2.0]]), b0=np.array([0.0]),
        W1=np.array([[3.0]]), b1=np.array([1.0]),
    )
    model = NpzSurrogate(str(path))
    np.testing.assert_allclose(
        model.predict(np.array([[0.5]])),
        [[3.0 * np.tanh(1.0) + 1.0]],
    )
