Portfolio view generator components and their interfaces.

- `view.py` -- `View` / `ViewSet`: the data VI-HRP actually consumes
  (`allocators/vihrp.py`). Node-targeted tilt + confidence, matched
  against a `ClusterTree` by its exact asset set.
- `generator.py` -- `ViewGenerator`: the `(returns_df, tree) -> ViewSet`
  shape any view-producing signal implements, so a new one can be
  dropped in without changing `allocators/vihrp.py` or callers wiring a
  generator into the backtest engine. Bind generator-specific
  hyperparameters (lookback, scale constants, ...) via
  `functools.partial` before passing the callable around.
- `momentum_generator.py` -- `generate_momentum_views`: the first
  `ViewGenerator`, a trailing-return momentum heuristic. Its `TILT_SCALE`
  and default `lookback` are calibrated against
  `data/processed/returns.parquet` via
  `scripts/calibrate_momentum_generator.py` (see that module's docstring
  for the tuning/validation split and results).
- `mean_reversion_generator.py` -- `generate_mean_reversion_views`: a
  second `ViewGenerator`, the mirror image of the momentum one (favors
  the LOWER-trailing-return side instead of the higher one). Motivated by
  `scripts/diagnose_regime.py`'s finding that this universe's own data
  shows weak reversal, not momentum. Neither of the first two generators
  beats plain HRP on this universe/period (see that module's docstring
  and `scripts/backtest_alternative_view_generators.py`'s committed
  output) -- both are documented, tested signals, not demonstrated
  improvements.
- `standardized_momentum_reversal_generator.py` --
  `generate_standardized_momentum_reversal_views`: a third
  `ViewGenerator`. The first two built their tilt from a raw,
  non-standardized, cluster-averaged return gap; this one computes a
  per-asset, volatility-standardized, cross-sectionally z-scored
  momentum+reversal combined signal instead (per
  Asness/Moskowitz/Pedersen 2013 and Baltas 2019), only aggregating to
  cluster level at the very end. `TILT_SCALE`/`REVERSAL_FORMATION_DAYS`
  are calibrated the same rigorous way as `momentum_generator.py`'s
  constants, via `scripts/calibrate_standardized_generator.py` -- and
  that calibration is itself the interesting (negative) result: every
  tested tilt strength underperforms simply turning the signal off
  (zero-view VI-HRP), so the "calibrated optimum" is really just
  "closest to no tilt at all." An uncalibrated, stronger version of this
  generator appeared to beat plain HRP in one sub-period; that result
  evaporated once calibrated properly from the whole in-sample window
  (see that module's docstring's "Calibration" section and
  `scripts/backtest_standardized_generator.py`'s committed output) --
  still net below plain HRP everywhere, same as the other two.
