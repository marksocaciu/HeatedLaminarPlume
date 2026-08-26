# LegacyPlume surrogate restart generation

`generate_initial_state.py` converts a learned, nondimensional surrogate into
the ordinary LegacyPlume checkpoint layout (`/mesh`, `/p_star`, `/u_star`, and
`/theta_star`). Run it in serial with the same FEniCS 2019.2 Python environment
as the solver.

The `.npz` must be pickle-free and contain:

- `feature_names`, for example `['x_star', 'y_star', 'Gr', 'Pr']`
- `target_names`, normally `['u_star', 'v_star', 'theta_star']`
- `x_mean`, `x_scale`, `y_mean`, `y_scale`
- either `coef`, `intercept` for a linear model
- or consecutive dense layers `W0`, `b0`, `W1`, `b1`, ... for a tanh MLP

Pressure is set to zero when it is not a learned target. The generated state is
clipped to `theta >= 0` by default, then passed through the solver's existing
boundary-condition routine.

Example:

```bash
python -m surrogate.generate_initial_state \
  --experiments-json experiments.json \
  --schema-json schema.json \
  --experiment-index 1 \
  --model surrogate/models/plume_surrogate.npz \
  --mesh-run-root PlumeCase/surrogate_mesh \
  --output PlumeCase/surrogate_initial_guess \
  --dt 1e-6
```

To reuse an already generated experiment mesh, replace `--mesh-run-root` with
the matching `--air-cells .../air_cells.xdmf --air-facets
.../air_facets.xdmf` pair. The input mesh is dimensional; the script scales it
in place by the experiment's existing `Lref` utility before inference/writing.

Start the normal solver with the resulting checkpoint directory:

```bash
python main.py --experiment-index 1 \
  --restart-from-checkpoint-mesh PlumeCase/surrogate_initial_guess
```
