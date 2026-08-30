import numpy as np
import warnings


class Stoch:
    "Stochastic Hydrological Models (Synthetic Time Series)"

    # Internal helpers
    @staticmethod
    def _is_stationary(ar_params):
        # Checks stationarity of an AR(p) process by verifying that the
        # roots of the characteristic polynomial 1 - phi_1*z - ... - phi_p*z^p
        # lie outside the unit circle.
        p = len(ar_params)
        if p == 0:
            return True
        # Companion-form characteristic polynomial coefficients (numpy convention:
        # highest degree first): z^p - phi_1*z^(p-1) - ... - phi_p
        coeffs = np.concatenate(([1.0], -np.array(ar_params, dtype=float)))
        roots = np.roots(coeffs)
        return np.all(np.abs(roots) > 1)

    @staticmethod
    def _combined_poly(params, seasonal_params, s, sign):
        # Builds the coefficient array (lag 0..order) of the product of a
        # non-seasonal lag polynomial and a seasonal lag polynomial, e.g.
        # (1 + sign*p1*B + sign*p2*B^2 + ...) * (1 + sign*P1*B^s + sign*P2*B^2s + ...)
        # Used to combine regular and seasonal AR/MA operators (Box-Jenkins
        # multiplicative form).
        ns_poly = np.zeros(len(params) + 1)
        ns_poly[0] = 1.0
        for i, val in enumerate(params):
            ns_poly[i + 1] = sign * val

        s_poly = np.zeros(len(seasonal_params) * s + 1)
        s_poly[0] = 1.0
        for j, val in enumerate(seasonal_params):
            s_poly[(j + 1) * s] = sign * val

        return np.convolve(ns_poly, s_poly)

    @staticmethod
    def _seasonal_cumsum(x, s):
        # Inverts a seasonal (lag-s) differencing operation: out[t] = out[t-s] + x[t]
        out = np.array(x, dtype=float).copy()
        for t in range(s, len(out)):
            out[t] += out[t - s]
        return out

    # Generative Models
    @staticmethod
    def ar_p(n_steps, mean, std_dev, ar_params, init_value=None):
        # Generate a synthetic series using a generic Autorregressive Model AR(p).
        # Generalizes the classic AR(1) formulation found in Bras & Rodriguez-Iturbe.

        # ar_params: list/array [phi_1, ..., phi_p], where phi_1 is the lag-1
        # coefficient, phi_2 the lag-2 coefficient, etc.
    
        # Note: unlike the closed-form AR(1) noise factor (std_dev*sqrt(1-rho^2)),
        # there is no simple closed-form variance correction for a general AR(p).
        # Instead, we simulate on a demeaned series driven by unit white noise and
        # rescale the result empirically to match the requested std_dev exactly
        # (same strategy already used in the arma() method).

        p = len(ar_params)

        if not Stoch._is_stationary(ar_params):
            warnings.warn(
                "ar_p: the given ar_params do not satisfy the stationarity "
                "condition (roots of the characteristic polynomial inside "
                "the unit circle). The series may diverge."
            )

        # 1. Extended array to handle historical lags during warm-up
        y = np.zeros(n_steps + p)
        wn = np.random.normal(0, 1, n_steps + p)

        # 2. Set initial boundary conditions (as anomalies from the mean) if provided
        if init_value is not None and len(init_value) >= p and p > 0:
            y[:p] = np.array(init_value[-p:]) - mean

        # 3. Core AR(p) simulation loop
        for t in range(p, n_steps + p):
            ar_term = sum(ar_params[i] * y[t - 1 - i] for i in range(p))
            y[t] = ar_term + wn[t]

        # 4. Slice off the warm-up period
        centered_series = y[p:]

        # 5. Empirical variance correction to match the requested std_dev exactly
        if np.std(centered_series) > 0:
            centered_series = (centered_series / np.std(centered_series)) * std_dev

        # 6. Add the historical mean back
        return centered_series + mean

    @staticmethod
    def ar_p_skewed(n_steps, mean, std_dev, ar_params, skewnesse, init_value=None):
        # Generate a synthetic series using a generic AR(p) model with skewed
        # (Wilson-Hilferty / Gamma-type) innovations.
        # Generalizes the classic AR(1)-skewed formulation found in
        # Bras & Rodriguez-Iturbe.

        # IMPORTANT / APPROXIMATION:
        # The original Bras & Rodriguez-Iturbe formula for gamma_e (the skew of
        # the innovations needed to obtain a target series skew) is derived
        # analytically for AR(1) only, as a function of the single lag-1
        # correlation coefficient. There is no simple closed-form equivalent for
        # a general AR(p). Here we approximate the "effective persistence" of
        # the process as sum(ar_params) and plug it into the same AR(1) formula.
        # This is a reasonable approximation for weakly-to-moderately correlated
        # processes, but the resulting series skewness will drift from the
        # requested `skewnesse` as p grows or as the AR structure becomes more
        # complex. Validate empirically (e.g. scipy.stats.skew) for your use case.

        p = len(ar_params)

        if not Stoch._is_stationary(ar_params):
            warnings.warn(
                "ar_p_skewed: the given ar_params do not satisfy the "
                "stationarity condition. The series may diverge."
            )

        y = np.zeros(n_steps + p)

        # 1. Standard Gaussian noise (mean=0, var=1)
        wn = np.random.normal(0, 1, n_steps + p)

        # 2. Effective persistence measure used as a stand-in for AR(1)'s rho
        rho_eff = sum(ar_params)

        # 3. Calculate Noise Skewness (Gamma E), approximated for AR(p)
        denom = (1 - rho_eff**2)
        if abs(denom) < 1e-9:
            gamma_e = skewnesse
        else:
            gamma_e = skewnesse * ((1 - rho_eff**3) / denom**1.5)

        # Avoid division by zero if data is perfectly symmetric
        if abs(gamma_e) < 10e-6:
            e = wn
        else:
            # 4. Wilson-Hilferty transformation to obtain Gamma Noise
            e = (2 / gamma_e) * ((1 + (gamma_e * wn / 6) - (gamma_e**2 / 36))**3) - (2 / gamma_e)

        # 5. Set initial boundary conditions if provided
        if init_value is not None and len(init_value) >= p and p > 0:
            y[:p] = np.array(init_value[-p:]) - mean

        # 6. Core AR(p) simulation loop using the transformed skewed noise (e)
        for t in range(p, n_steps + p):
            ar_term = sum(ar_params[i] * y[t - 1 - i] for i in range(p))
            y[t] = ar_term + e[t]

        centered_series = y[p:]

        # 7. Empirical variance correction
        if np.std(centered_series) > 0:
            centered_series = (centered_series / np.std(centered_series)) * std_dev

        return centered_series + mean

    @staticmethod
    def arma(n_steps, mean, std_dev, ar_params, ma_params, init_value=None):
        # Generate a synthetic series using the ARMA (p, q) Model
        # Bases on the classic Box-Jenkins formulation

        # 1. Extract the orders "p" (AR lags) and 'q' (MA lags)
        p, q = len(ar_params), len(ma_params)
        max_lag = max(p, q)

        # 2. Initialize extended arrays to handle historical lags during warm-up
        y = np.zeros(n_steps + max_lag)
        w = np.random.normal(0, 1, n_steps + max_lag)

        # 3. Set initial boundary conditions for anomalies if provided
        if init_value is not None and len(init_value) >= p and p > 0:
            y[max_lag - p:max_lag] = np.array(init_value[-p:]) - mean

        # 4. Core ARMA simulation loop (starts after the max lag offset)
        for t in range(max_lag, n_steps + max_lag):
            ar_term = sum(ar_params[i] * y[t - 1 - i] for i in range(p))
            ma_term = sum(ma_params[j] * w[t - 1 - j] for j in range(q))
            y[t] = ar_term + w[t] + ma_term

        # 5. Slice the array to remove the initial warm-up/lag offset period
        centered_series = y[max_lag:]

        # 6. Empirical variance correction
        # Mathematical interaction between AR and MA parameters distorts original variance
        # We rescale the series to match the exact historical standard deviation requested
        if np.std(centered_series) > 0:
            centered_series = (centered_series / np.std(centered_series)) * std_dev

        # 7. Add the historical mean back to shift the series from anomalies to real values
        return centered_series + mean

    @staticmethod
    def sarima(n_steps, mean, std_dev, ar_params, ma_params,
               seasonal_ar_params, seasonal_ma_params, seasonal_period,
               d=0, D=0, init_value=None):
    
        s = seasonal_period
        p, q = len(ar_params), len(ma_params)
        P, Q = len(seasonal_ar_params), len(seasonal_ma_params)

        # 1. Combine regular and seasonal AR / MA operators (Box-Jenkins
        # multiplicative polynomial expansion: phi(B)*Phi(B^s), theta(B)*Theta(B^s))
        ar_full = Stoch._combined_poly(ar_params, seasonal_ar_params, s, sign=-1)
        ma_full = Stoch._combined_poly(ma_params, seasonal_ma_params, s, sign=1)

        ar_order = len(ar_full) - 1  # = p + P*s
        ma_order = len(ma_full) - 1  # = q + Q*s
        max_lag = max(ar_order, ma_order, 1)

        if ar_order > 0 and not Stoch._is_stationary(-ar_full[1:ar_order + 1]):
            warnings.warn(
                "sarima: the combined AR/seasonal-AR parameters do not satisfy "
                "the stationarity condition. The stationary component may diverge."
            )

        # 2. Initialize extended arrays to handle historical lags during warm-up
        y = np.zeros(n_steps + max_lag)
        w = np.random.normal(0, 1, n_steps + max_lag)

        # 3. Set initial boundary conditions for anomalies if provided
        if init_value is not None and len(init_value) >= ar_order and ar_order > 0:
            y[max_lag - ar_order:max_lag] = np.array(init_value[-ar_order:]) - mean

        # 4. Core multiplicative seasonal ARMA simulation loop
        for t in range(max_lag, n_steps + max_lag):
            ar_term = -sum(ar_full[k] * y[t - k] for k in range(1, ar_order + 1)) if ar_order > 0 else 0.0
            ma_term = sum(ma_full[k] * w[t - k] for k in range(1, ma_order + 1)) if ma_order > 0 else 0.0
            y[t] = ar_term + w[t] + ma_term

        # 5. Slice off the warm-up period
        stationary_series = y[max_lag:]

        # 6. Empirical variance correction on the stationary component
        if np.std(stationary_series) > 0:
            stationary_series = (stationary_series / np.std(stationary_series)) * std_dev

        # 7. Integrate: invert seasonal differencing D times, then regular differencing d times
        integrated = stationary_series
        for _ in range(D):
            integrated = Stoch._seasonal_cumsum(integrated, s)
        for _ in range(d):
            integrated = np.cumsum(integrated)

        # 8. Add the mean/baseline level back
        return integrated + mean