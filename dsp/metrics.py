"""Evaluation metrics for pitch estimation (exact definitions from the guide).

All frequency inputs are in Hz. A missing estimate (rejected frame) is
represented by None or NaN and is never counted as an error value.
"""
import numpy as np

GROSS_THRESHOLD_CENTS = 50.0


def _to_float_array(values):
    """Convert a list/array that may contain None into a float array (None -> NaN)."""
    return np.array(
        [np.nan if v is None else v for v in np.atleast_1d(values)],
        dtype=float,
    )


def cents_error(f_est, f_true):
    """Cents error = 1200 * log2(f_est / f_true).

    Returns NaN where the estimate is missing or non-positive.
    f_true may be a scalar or an array matching f_est.
    """
    est = _to_float_array(f_est)
    true = np.broadcast_to(np.asarray(f_true, dtype=float), est.shape)

    out = np.full(est.shape, np.nan)
    ok = np.isfinite(est) & (est > 0) & np.isfinite(true) & (true > 0)
    out[ok] = 1200.0 * np.log2(est[ok] / true[ok])
    return out


def octave_error_mask(errors_cents, tol_cents=GROSS_THRESHOLD_CENTS):
    """True where the estimate is about a factor of 2 high or low (+-1200 cents)."""
    e = np.asarray(errors_cents, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.isfinite(e) & (np.abs(np.abs(e) - 1200.0) <= tol_cents)


def gross_error_mask(errors_cents, threshold=GROSS_THRESHOLD_CENTS):
    """True where |error| > threshold cents. Octave errors are always included,
    because they are ~1200 cents off."""
    e = np.asarray(errors_cents, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.isfinite(e) & (np.abs(e) > threshold)


def rmse_cents(errors_cents):
    """Root mean square of the cents error over accepted (finite) frames only.
    Returns NaN if there are none. Always report next to gross_error_rate."""
    e = np.asarray(errors_cents, dtype=float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(e ** 2)))


def gross_error_rate(errors_cents, threshold=GROSS_THRESHOLD_CENTS):
    """Percent of accepted frames with |error| > threshold (or octave error)."""
    e = np.asarray(errors_cents, dtype=float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return float("nan")
    return float(100.0 * np.mean(gross_error_mask(e, threshold)))


def octave_error_rate(errors_cents):
    """Percent of accepted frames that are octave errors (for E9)."""
    e = np.asarray(errors_cents, dtype=float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return float("nan")
    return float(100.0 * np.mean(octave_error_mask(e)))


def voiced_accepted_rate(accepted, truly_voiced):
    """Fraction of truly voiced frames that were accepted."""
    accepted = np.asarray(accepted, dtype=bool)
    voiced = np.asarray(truly_voiced, dtype=bool)
    if accepted.shape != voiced.shape:
        raise ValueError("accepted and truly_voiced must have the same shape.")
    n = voiced.sum()
    return float("nan") if n == 0 else float((accepted & voiced).sum() / n)


def silence_falsely_accepted(accepted, truly_voiced):
    """Fraction of silent frames that were wrongly accepted."""
    accepted = np.asarray(accepted, dtype=bool)
    voiced = np.asarray(truly_voiced, dtype=bool)
    if accepted.shape != voiced.shape:
        raise ValueError("accepted and truly_voiced must have the same shape.")
    silent = ~voiced
    n = silent.sum()
    return float("nan") if n == 0 else float((accepted & silent).sum() / n)


def summarize(f_est, f_true, truly_voiced=None):
    """One-call summary used by every experiment so all of them report the same
    numbers: RMSE, gross error rate, octave error rate and accepted fraction."""
    err = cents_error(f_est, f_true)
    accepted = np.isfinite(err)
    out = {
        "rmse_cents": rmse_cents(err),
        "gross_error_pct": gross_error_rate(err),
        "octave_error_pct": octave_error_rate(err),
        "accepted_frac": float(accepted.mean()) if accepted.size else float("nan"),
    }
    if truly_voiced is not None:
        out["voiced_accepted"] = voiced_accepted_rate(accepted, truly_voiced)
        out["silence_false_accept"] = silence_falsely_accepted(accepted, truly_voiced)
    return out


def latency_stats(latencies_ms, min_trials=10):
    """Mean and sample standard deviation (ms). Needs at least min_trials values."""
    x = np.asarray(latencies_ms, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < min_trials:
        raise ValueError(f"Need at least {min_trials} latency trials, got {x.size}.")
    return float(np.mean(x)), float(np.std(x, ddof=1))
