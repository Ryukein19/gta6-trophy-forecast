"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts every trophy's Exophase earn rate at a case's target. The model
is an earners / players decomposition, fitted the way the counts actually
behave:

  - The players tracked N(t) are fitted with a saturating (logistic) curve
    in log-time. New players dilute every rate, so the denominator's growth
    is the dilution scale; the implied earners at the target are
    rate * N, which keeps the forecast count-consistent (earners <= players).
  - Each trophy's RATE is fitted with a saturating curve in log-time. The
    rate is the quantity scored, and it is the saturating curve: earners
    are players x progress, a product of two logistics, so fitting earners
    as a single logistic is misspecified and its extrapolation runs the
    wrong way. Fitting the rate directly captures the net of earning flow
    (players progressing push it up) against dilution (new players push it
    down), which is what the history shows.
  - Two guards keep a far extrapolation honest. The rate's asymptote is
    capped: it cannot exceed the observed range by more than twice the
    observed movement (and never 100), so a trophy still rising steeply
    cannot be forecast at 99.99. And the extrapolation is damped toward
    the last observed rate in logit space, weighted by how far the target
    sits beyond the observed log-time range.
  - Quantiles are centred on the forecast rate with a spread that grows
    with the case's horizon: a random-walk floor plus the fit's own
    residual uncertainty, widened the further the target sits beyond the
    observed range. A spread that is too wide pays at every trophy and
    one too narrow pays for every trophy outside it, so the spread is
    kept honest, not maximal.

Python 3.13, standard library only. No network. Run as:
    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves this file's directory off sys.path; put it back
# first, so a helper module shipped beside this file would import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Random-walk spread floor, in logit units per square-root day. Smaller
# than the house baseline's 0.5: the model already carries the systematic
# drift, so only the residual walk around the forecast remains.
RW_SIGMA = 0.3
# Widest spread allowed, in logit units (keeps a bad fit from exploding).
SIGMA_CAP = 2.5
# Smallest model spread, in logit units (even a perfect fit is uncertain).
SIGMA_FLOOR = 0.05


def clamp_pct(x: float) -> float:
    """A percent into the range the logit is defined on."""
    return min(99.99, max(0.01, x))


def logit_of(pct: float) -> float:
    """logit of a percent, clamped to 0.01..99.99 first."""
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def pct_of_logit(value: float) -> float:
    """A logit back to a percent, clamped to the defined range."""
    if value >= 40.0:
        return 99.99
    if value <= -40.0:
        return 0.01
    return clamp_pct(100.0 / (1.0 + math.exp(-value)))


def round4(x: float):
    """Round to 4 decimals, half up, as the grader's JavaScript does.

    A whole number comes back as an int, so json.dumps writes 100 rather
    than 100.0.
    """
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text: str) -> float:
    """An ISO-8601 timestamp to epoch seconds."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def log_time(epoch: float, origin: float) -> float:
    """Log-time: ln(1 + hours since the origin)."""
    hours = (epoch - origin) / 3600.0
    return math.log1p(max(0.0, hours))


def _ols(xs, ys):
    """Least-squares line y = b0 + b1*x; returns (slope, intercept)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return 0.0, my
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


def fit_saturating(xs, ys, a_cap=None):
    """Fit y = A / (1 + exp(-(b0 + b1*x))) to the points (xs, ys).

    The asymptote A is found by a geometric grid search: for a fixed A the
    logit of y/A is linear in x, so b0 and b1 come from ordinary least
    squares, and the asymptote with the smallest squared error is kept.
    a_cap, when given, bounds the asymptote from above: a series that is
    still rising steeply has many asymptotes that fit it, and an unbounded
    one can run away, so the cap keeps the ceiling plausible.

    Returns a dict with the fitted parameters and the residual scatter in
    logit units, or None when there are too few positive points to fit.
    """
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None and y > 0.0]
    if len(pts) < 3:
        return None
    px = [p[0] for p in pts]
    py = [p[1] for p in pts]
    top = max(py)

    hi = top * 40.0
    if a_cap is not None:
        hi = min(hi, max(top * 1.05, a_cap))
    grid = []
    a = top * 1.05
    while a <= hi:
        grid.append(a)
        a *= 1.35
    if not grid:
        grid = [top * 1.05]

    scored = []
    for a in grid:
        ly = [math.log((y / a) / (1.0 - y / a)) for y in py]
        slope, intercept = _ols(px, ly)
        sse = sum((v - (intercept + slope * x)) ** 2 for x, v in zip(px, ly))
        scored.append((sse, a, intercept, slope))

    sse, a, intercept, slope = min(scored, key=lambda s: s[0])
    rmse = math.sqrt(sse / len(pts))
    return {
        "asymptote": a,
        "intercept": intercept,
        "slope": slope,
        "rmse_logit": rmse,
        "n": len(pts),
        "x_first": px[0],
        "x_last": px[-1],
    }


def predict(fit, x: float):
    """A fitted saturating curve evaluated at log-time x."""
    if fit is None:
        return None
    z = fit["intercept"] + fit["slope"] * x
    if z > 700.0:
        return fit["asymptote"]
    if z < -700.0:
        return 0.0
    return fit["asymptote"] / (1.0 + math.exp(-z))


def _origin_epoch(case: dict, history: list) -> float:
    """The time origin for log-time: the game's release, else the first read."""
    game = case.get("game") or {}
    release = game.get("release_date")
    if release:
        try:
            return parse_time(release + "T00:00:00Z")
        except ValueError:
            pass
    return parse_time(history[0]["at"])


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_days = max(0.0, case["horizon_minutes"] / 1440.0)

    origin = _origin_epoch(case, history)
    target_x = log_time(parse_time(case["target"]), origin)

    # --- players tracked: the dilution scale, one point per read -------
    read_x = []
    read_n = []
    for read in history:
        n = read.get("players_tracked")
        if n is None:
            continue
        read_x.append(log_time(parse_time(read["at"]), origin))
        read_n.append(float(n))
    if not read_x:
        raise ValueError(f"{name} has no player counts")
    last_n = read_n[-1]

    players_fit = fit_saturating(read_x, read_n)
    n_target = predict(players_fit, target_x)
    if n_target is None or n_target <= 0.0:
        n_target = last_n
    # Dilution uncertainty: how well the players curve tracks, in log
    # space. When the player growth itself is uncertain, the dilution
    # the rate will suffer is uncertain too, so the spread widens.
    players_resid = []
    if players_fit is not None:
        for xj, nj in zip(read_x, read_n):
            fitted_n = predict(players_fit, xj)
            if fitted_n is None or fitted_n <= 0.0 or nj <= 0.0:
                continue
            players_resid.append(math.log(nj) - math.log(fitted_n))
    players_rmse = (
        math.sqrt(sum(r * r for r in players_resid) / len(players_resid))
        if players_resid
        else 0.25
    )

    # Random-walk floor for the spread, scaled to this case's horizon.
    sigma_rw = RW_SIGMA * math.sqrt(horizon_days)
    # How far the target sits beyond the observed log-time range; the
    # damping weight and the spread both grow with it.
    x_span = max(0.25, read_x[-1] - read_x[0])
    reach = max(0.0, target_x - read_x[-1])
    damp = x_span / (x_span + reach)
    spread_reach = 1.0 + min(2.0, reach / x_span)

    rows = []
    for i, trophy in enumerate(trophies):
        # --- this trophy's rate over time, one point per read ----------
        rx = []
        ry = []
        last_rate = None
        for read in history:
            rates = read.get("rates") or []
            rate = rates[i] if i < len(rates) else None
            if rate is not None:
                last_rate = rate
            if rate is None or rate <= 0.0:
                continue
            rx.append(log_time(parse_time(read["at"]), origin))
            ry.append(float(rate))

        if not ry or last_rate is None:
            # Never earned (or never read): hold the floor, widen the band.
            median = 0.01
            sigma_model = 1.0
        else:
            # The rate's asymptote cannot exceed the observed range by more
            # than twice the observed movement, and never 100.
            span = max(ry) - min(ry)
            cap = min(99.0, max(ry) * 1.02 + 2.0 * span)
            # Rare trophies: the percent-space cap above is far too tight
            # in logit terms. A rate of 0.8% capped at ~1.4% cannot rise
            # by even one logit unit, yet a late-unlocking rare trophy can
            # rise several logit units over a multi-day horizon (players
            # reach it late, after the observed window). Give low-rate
            # trophies a logit-space floor on the cap so the fitted
            # asymptote is allowed a meaningful rise: twice the observed
            # logit movement plus a horizon allowance. This only raises
            # the ceiling the grid search may choose; the fit still picks
            # the asymptote that best fits the history.
            if max(ry) < 5.0:
                lmax = logit_of(max(ry))
                lmin = logit_of(min(ry))
                lspan = max(0.25, lmax - lmin)
                margin = 2.0 * lspan + 0.5 * math.sqrt(horizon_days)
                cap_logit = min(logit_of(99.0), lmax + margin)
                cap = max(cap, pct_of_logit(cap_logit))
            rate_fit = fit_saturating(rx, ry, a_cap=cap)
            r_target = predict(rate_fit, target_x) if rate_fit else None
            if r_target is None:
                median = clamp_pct(last_rate)
                sigma_model = 1.0
            else:
                # Damp the extrapolation toward the last observed rate.
                # Rare trophies: the last observed rate is a poor anchor
                # (the rate is about to rise as players reach the trophy,
                # so the tiny last rate drags the forecast down). Give the
                # fitted curve more weight for low-rate trophies by
                # halving the effective reach in the damping.
                trophy_damp = damp
                if max(ry) < 5.0:
                    trophy_damp = x_span / (x_span + reach * 0.5)
                blended = (
                    trophy_damp * logit_of(r_target)
                    + (1.0 - trophy_damp) * logit_of(last_rate)
                )
                median = clamp_pct(pct_of_logit(blended))
                # Residual scatter of the fitted rate against the observed
                # one, in logit units: how far the model already misses
                # in-sample.
                resid = []
                if rate_fit is not None:
                    for xj, yj in zip(rx, ry):
                        fitted = predict(rate_fit, xj)
                        if fitted is None or fitted <= 0.0:
                            continue
                        resid.append(logit_of(yj) - logit_of(fitted))
                if len(resid) >= 2:
                    sigma_model = math.sqrt(
                        sum(r * r for r in resid) / len(resid)
                    )
                elif resid:
                    sigma_model = abs(resid[0])
                else:
                    sigma_model = 0.25

        # Count-consistency: the implied earners at the target are
        # median / 100 * n_target, and because the median is capped below
        # 100 they can never exceed the players then. No separate clamp
        # is needed; the rate cap above is the guarantee.

        sigma = math.sqrt(
            sigma_rw ** 2
            + (max(SIGMA_FLOOR, sigma_model) * spread_reach) ** 2
            + (players_rmse * spread_reach) ** 2
        )
        sigma = min(SIGMA_CAP, sigma)

        centre = logit_of(median)
        qs = [round4(pct_of_logit(centre + z * sigma)) for z in Z]
        # Keep the row non-decreasing after rounding.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
