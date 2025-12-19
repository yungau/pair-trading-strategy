import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def apply_position_filter(signal: pd.Series, filter_ts: pd.Series) -> pd.Series:
    """
    Vectorized filter with 'wait-for-next-change' semantics:
      - Filter True => flat *at that timestamp* (force unwind / ignore entry).
      - After a filter-forced unwind, wait until the *next change* in the original signal,
        then allow that entry even if filter is still True (one-time override).
      - If a new entry was blocked by the filter while flat, also wait until the next change before entering.
      - Exit immediately when original signal == 0.
    """
    signal = signal.copy()
    filter_ts = filter_ts.reindex(signal.index).fillna(False)

    # Original change events (entry/switch/exit): only where the original signal changes
    prev_signal = signal.shift(1).fillna(0)
    change = signal.where(signal != prev_signal, 0)

    # Baseline used only to detect whether we had a position before (ignores filter)
    pos_base = change.replace(0, np.nan).ffill().fillna(0)
    pos_base = pos_base.mask(signal == 0, 0)

    # Episode starts:
    # - unwind_points: filter True while we *had* a position (forced exit)
    # - block_entry_points: filter True while flat but original wants to enter now
    had_pos_before = pos_base.shift(1).fillna(0) != 0
    unwind_points = filter_ts & had_pos_before
    block_entry_points = filter_ts & (~had_pos_before) & (pos_base != 0)

    # Defer segments start AFTER the episode timestamp (shifted)
    defer_seg = (unwind_points | block_entry_points).shift().fillna(False).cumsum()

    # First subsequent change AFTER a defer episode (unwind or block)
    cond_any = (defer_seg > 0) & (change != 0)
    entry_count_any = cond_any.groupby(defer_seg).cumsum()
    # first_entry_any = cond_any & (entry_count_any == 1)

    # Waiting until BEFORE the first subsequent change
    waiting_any = (defer_seg > 0) & (entry_count_any == 0)

    # For the 'regardless filter' exception: first entry after UNWIND episodes only
    unwind_seg = unwind_points.shift().fillna(False).cumsum()
    cond_unwind = (unwind_seg > 0) & (change != 0)
    entry_count_unwind = cond_unwind.groupby(unwind_seg).cumsum()
    first_entry_after_unwind = cond_unwind & (entry_count_unwind == 1)

    # Allowed entry (±1) timestamps: change is present and either filter is False,
    # or this is the first entry after an UNWIND (override), and we are not waiting
    allowed_entry = (change != 0) & ((~filter_ts) | first_entry_after_unwind) & (~waiting_any)

    # Explicit resets to flat at timestamps where:
    #  (a) original signal is 0, or
    #  (b) filter True (except the allowed-entry override), or
    #  (c) we are in the waiting window
    reset_flat = (signal == 0) | (filter_ts & ~allowed_entry) | waiting_any

    # Build position stream:
    #  - 0 where reset (breaks ffill)
    #  - ±1 where allowed entry
    #  - NaN otherwise (carry prior state)
    raw = np.where(reset_flat, 0, np.where(allowed_entry, signal, np.nan))
    position = pd.Series(raw, index=signal.index).ffill().fillna(0)

    return pd.Series(position, index=signal.index)


class RegimeShiftDetector:
    def __init__(self,
                 ref_window: int = 126,
                 test_window: int = 20,
                 gap: int = 1,
                 threshold_quantile: float = 0.97):  
        self.ref_window = int(ref_window)
        self.test_window = int(test_window)
        self.gap = int(gap)
        self.threshold_quantile = threshold_quantile
        self._gamma = None
        self._threshold = None
        self.rng = np.random.default_rng(42)

    def _median_heuristic(self, X):
        n = min(1000, len(X))
        idx = self.rng.choice(len(X), size=n, replace=False)
        Xsub = X[idx]
        sq_dists = ((Xsub[:, None, :] - Xsub[None, :, :]) ** 2).sum(axis=2)
        sq_dists = sq_dists[np.triu_indices(n, k=1)]
        sq_dists = sq_dists[sq_dists > 0]
        median_sq = np.median(sq_dists) if len(sq_dists) > 0 else 1.0
        return 1.0 / (2 * median_sq)

    def _mmd2_u_statistic(self, X, Y):
        n, m = len(X), len(Y)
        if n < 2 or m < 2:
            return np.nan
        
        def k(A, B):
            return np.exp(-self._gamma * ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
        
        Kxx = k(X, X); np.fill_diagonal(Kxx, 0)
        Kyy = k(Y, Y); np.fill_diagonal(Kyy, 0)
        Kxy = k(X, Y)
        h11 = Kxx.sum() / (n * (n - 1))
        h22 = Kyy.sum() / (m * (m - 1))
        h12 = Kxy.sum() / (n * m)
        return h11 + h22 - 2 * h12

    def run(self, features_df: pd.DataFrame, zscore_series: pd.Series = None) -> pd.DataFrame:
        features_df = features_df.ffill().bfill()
        X = features_df.values.astype(float)
        dates = features_df.index
        N = len(X)

        self._gamma = self._median_heuristic(X)

        mmd2_series = np.full(N, np.nan)
        threshold_series = np.full(N, np.nan)
        shift_mmd = np.full(N, False)

        start_idx = self.ref_window + self.gap + self.test_window - 1
        historical_mmds = []

        for i in range(start_idx, N):
            ref_end = i - self.test_window - self.gap + 1
            ref_start = ref_end - self.ref_window
            test_start = ref_end + self.gap
            test_end = i + 1

            ref_data = X[ref_start:ref_end]
            test_data = X[test_start:test_end]

            if len(ref_data) != self.ref_window or len(test_data) != self.test_window:
                continue

            mmd2 = self._mmd2_u_statistic(ref_data, test_data)
            mmd2_series[i] = mmd2
            historical_mmds.append(mmd2)

            # Rolling adaptive threshold (last ref_window values)
            if len(historical_mmds) >= self.ref_window:
                recent = np.array(historical_mmds)[-self.ref_window:]
                threshold = np.quantile(recent, self.threshold_quantile)
                threshold_series[i] = threshold
                shift_mmd[i] = mmd2 > threshold

        regime_shift = shift_mmd 

        result = pd.DataFrame({
            'regime_shift': regime_shift,
            'mmd2': mmd2_series,
            'threshold': threshold_series
        }, index=dates)

        return result


