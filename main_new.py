# inflation_hedge_optimizer_app.py
# Streamlit app: Inflation "floor" (CPI), risk-free comparator, and portfolio optimizer with drawdown constraint
# Date: 2025-10-28

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # <-- ensures headless rendering in Streamlit
import matplotlib.pyplot as plt

# ---------- Page config ----------
st.set_page_config(
    page_title="Inflation‑Hedge Optimizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Inflation‑Hedge Optimizer")
st.caption("Floor = CPI (flat in real terms) • Compare to a smooth risk‑free line • Optimize by Sharpe, Sortino, Sterling, Calmar, or minimize CVaR • Optional max drawdown constraint")

# ---------- Helper functions ----------
def to_monthly_last(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("M").last()

def cumulative_wealth(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).cumprod()

def drawdown(wealth: pd.Series) -> pd.Series:
    peak = wealth.cummax()
    return wealth / peak - 1.0

def max_drawdown(returns: pd.Series) -> float:
    dd = drawdown(cumulative_wealth(returns))
    return float(dd.min())  # negative number

def annualize_mean_std(ret_m: pd.Series) -> tuple[float, float]:
    mu_a = ret_m.mean() * 12.0
    sig_a = ret_m.std(ddof=0) * np.sqrt(12.0)
    return float(mu_a), float(sig_a)

def cagr_from_wealth(wealth: pd.Series) -> float:
    n_months = wealth.shape[0]
    if n_months <= 1:
        return 0.0
    years = n_months / 12.0
    return float(wealth.iloc[-1]**(1/years) - 1.0)

def portfolio_series(weights: np.ndarray, rets: pd.DataFrame) -> pd.Series:
    w = np.asarray(weights).reshape(-1,)
    return (rets @ w).rename("Portfolio")

def sortino_ratio(w: np.ndarray, rets: pd.DataFrame, rf_m: float = 0.0) -> float:
    pr = portfolio_series(w, rets)
    excess = pr - rf_m
    downside = excess.copy()
    downside[downside > 0.0] = 0.0
    dd_std = np.sqrt((downside**2).mean())
    if dd_std == 0:
        return np.inf
    mu_ex_a = excess.mean() * 12.0
    dd_std_a = dd_std * np.sqrt(12.0)
    return mu_ex_a / dd_std_a

def sharpe_ratio(w: np.ndarray, rets: pd.DataFrame, rf_m: float = 0.0) -> float:
    pr = portfolio_series(w, rets)
    excess = pr - rf_m
    mu_ex_a = excess.mean() * 12.0
    sig_a = pr.std(ddof=0) * np.sqrt(12.0)
    if sig_a == 0:
        return np.inf
    return mu_ex_a / sig_a

def calmar_ratio(w: np.ndarray, rets: pd.DataFrame) -> float:
    pr = portfolio_series(w, rets)
    wealth = cumulative_wealth(pr)
    cagr = cagr_from_wealth(wealth)
    mdd = abs(min(0.0, max_drawdown(pr)))
    if mdd == 0:
        return np.inf
    return cagr / mdd

def sterling_ratio(w: np.ndarray, rets: pd.DataFrame) -> float:
    # Approximation: CAGR divided by average of the 3 deepest drawdowns
    pr = portfolio_series(w, rets)
    wealth = cumulative_wealth(pr)
    dd = drawdown(wealth)  # negative values
    local_mins = dd.cummin()
    events = local_mins.where(local_mins.shift(1) > local_mins).dropna()
    if len(events) == 0:
        denom = 1e-9
    else:
        worst3 = np.sort(events.values)[:3]  # the 3 most negative
        denom = np.mean(np.abs(worst3))
        if denom == 0:
            denom = 1e-9
    cagr = cagr_from_wealth(wealth)
    return cagr / denom

def cvar_loss(w: np.ndarray, rets: pd.DataFrame, alpha: float = 0.05) -> float:
    pr = portfolio_series(w, rets)
    pr_sorted = np.sort(pr.values)  # ascending
    n = len(pr_sorted)
    k = max(1, int(np.ceil(alpha * n)))
    tail = pr_sorted[:k]
    # CVaR is (negative) expected loss in the alpha tail
    return float(-np.mean(tail))

def weight_constraints(n: int, enforce_dd: bool, rets: pd.DataFrame, dd_limit: float):
    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # sum to 1
    ]
    if enforce_dd:
        def dd_con(w):
            pr = portfolio_series(w, rets)
            mdd = abs(min(0.0, max_drawdown(pr)))  # positive
            return dd_limit - mdd  # keep <= dd_limit -> inequality >= 0
        cons.append({"type": "ineq", "fun": dd_con})
    return cons

def optimize(rets: pd.DataFrame, method: str, rf_m: float, enforce_dd: bool, dd_limit: float) -> np.ndarray:
    n = rets.shape[1]
    x0 = np.array([1.0/n]*n)
    bounds = [(0.0, 1.0)] * n
    cons = weight_constraints(n, enforce_dd, rets, dd_limit)

    if method == "Sharpe":
        obj = lambda w: -sharpe_ratio(w, rets, rf_m)
    elif method == "Sortino":
        obj = lambda w: -sortino_ratio(w, rets, rf_m)
    elif method == "Sterling":
        obj = lambda w: -sterling_ratio(w, rets)
    elif method == "Calmar":
        obj = lambda w: -calmar_ratio(w, rets)
    elif method == "CVaR (minimize)":
        obj = lambda w: cvar_loss(w, rets, alpha=0.05)
    else:
        obj = lambda w: -sharpe_ratio(w, rets, rf_m)

    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    if not res.success:
        st.warning(f"Optimizer did not fully converge ({res.message}). Using best found weights.")
    w = res.x
    # numerical cleanup
    w[w < 1e-8] = 0.0
    if w.sum() == 0:
        w = np.array([1.0/n]*n)
    else:
        w = w / w.sum()
    return w

def load_prices(tickers, start):
    # yfinance download (Adj Close), monthly last
    data = yf.download(tickers, start=start, auto_adjust=False, progress=False)["Adj Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.ffill()
    data = to_monthly_last(data)
    return data


import io, requests
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st

@st.cache_data(ttl=86400, show_spinner=False)
def load_cpi(start: datetime) -> pd.DataFrame:
    """
    Fetch CPI (CPIAUCSL) monthly levels from FRED with robust fallbacks.
    Works on Python 3.12/3.13 and handles occasional non-CSV responses.
    """
    urls = [
        # Static “downloaddata” CSV (no API key, full history)
        "https://fred.stlouisfed.org/series/CPIAUCSL/downloaddata/CPIAUCSL.csv",
        # Graph CSV with start filter (fallback)
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL&observation_start={start:%Y-%m-%d}",
    ]
    headers = {"User-Agent": "Mozilla/5.0 (Streamlit app)"}

    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            # Parse as CSV (skip comment lines if present)
            df = pd.read_csv(io.StringIO(r.text), parse_dates=["DATE"], comment="#")
            # Some endpoints provide VALUE instead of CPIAUCSL
            value_col = "CPIAUCSL" if "CPIAUCSL" in df.columns else ("VALUE" if "VALUE" in df.columns else None)
            if value_col is None:
                continue  # not a valid CSV, try next URL

            df = (df[["DATE", value_col]]
                    .rename(columns={"DATE": "Date", value_col: "CPI"}))
            df["CPI"] = pd.to_numeric(df["CPI"], errors="coerce")
            df = df.dropna(subset=["CPI"]).sort_values("Date")
            df = df[df["Date"] >= pd.Timestamp(start)]
            df = df.set_index("Date").resample("M").last()
            if not df.empty:
                return df
        except Exception as e:
            st.warning(f"FRED fetch failed for {url}: {e}")

    # Last-resort fallback so the app stays usable
    st.warning("CPI unavailable; using a flat CPI baseline temporarily.")
    idx = pd.date_range(start, pd.Timestamp.today(), freq="M")
    return pd.DataFrame({"CPI": np.full(len(idx), 100.0)}, index=idx)



def align_and_trim(prices: pd.DataFrame, cpi: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Align monthly dates and drop rows/cols that are all-NaN
    df = prices.copy()
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    cpi = cpi.reindex(df.index).ffill().dropna()
    # Align again by intersection
    idx = df.index.intersection(cpi.index)
    return df.loc[idx], cpi.loc[idx]

def to_real_returns(rets_nominal: pd.DataFrame, cpi: pd.DataFrame) -> pd.DataFrame:
    # cpi is level index (monthly), convert to inflation rate per month
    infl = cpi["CPI"].pct_change().reindex(rets_nominal.index).fillna(0.0)
    real = ((1.0 + rets_nominal).div(1.0 + infl, axis=0) - 1.0)
    return real

def riskfree_wealth(index: pd.Index, rf_ann: float) -> pd.Series:
    rf_m = (1.0 + rf_ann)**(1.0/12.0) - 1.0
    steps = np.arange(len(index))
    return pd.Series((1.0 + rf_m)**steps, index=index, name="Risk‑Free")

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("⚙️ Controls")

    # Universe with 15+ years history
    universe = {
        "SPY": "US Stocks (S&P 500)",
        "VEA": "Developed ex‑US Stocks",
        "EEM": "Emerging Mkts Stocks",
        "GLD": "Gold",
        "VNQ": "US REITs",
        "LQD": "IG Corporate Bonds",
        "IEF": "7–10Y Treasuries",
        "TLT": "20+Y Treasuries",
        "AGG": "US Aggregate Bond",
        "DBC": "Broad Commodities",
        "BTC-USD": "Bitcoin (Price History)",
    }

    tickers_default = ["SPY", "VEA", "IEF", "GLD", "VNQ", "DBC", "BTC-USD"]
    tickers = st.multiselect("Asset universe (15+ yrs):", options=list(universe.keys()), default=tickers_default, format_func=lambda t: f"{t} — {universe[t]}")
    if len(tickers) < 2:
        st.warning("Select at least two assets to optimize.")
    start_year = st.slider("Start year", min_value=2005, max_value=datetime.today().year-1, value=2010, step=1)
    start = datetime(start_year, 1, 1)

    st.divider()

    rf_ann = st.slider("Risk‑free annual rate (smooth comparator)", min_value=0.0, max_value=0.10, value=0.03, step=0.005)
    plot_real = st.checkbox("Plot in **real** terms (CPI‑adjusted) — CPI becomes a flat floor", value=True)

    st.divider()

    method = st.selectbox("Optimize", options=["Sharpe", "Sortino", "Sterling", "Calmar", "CVaR (minimize)"])

    enforce_dd = st.checkbox("Enforce Max Drawdown constraint", value=False)
    dd_limit = st.slider("Max Drawdown limit", min_value=0.05, max_value=0.80, value=0.25, step=0.05)

    st.caption("Tip: Turn **on** the constraint to find the best return subject to the chosen max drawdown bound.")

# ---------- Data ----------
if len(tickers) >= 2:
    with st.spinner("Downloading data…"):
        prices = load_prices(tickers, start)
        cpi = load_cpi(start)

    # Keep intersection and monthly
    prices, cpi = align_and_trim(prices, cpi)

    # Compute monthly returns
    rets_nom = prices.pct_change().dropna(how="all").dropna(axis=1)

    # Optional: real returns (CPI‑adjusted so CPI is flat)
    if plot_real:
        rets = to_real_returns(rets_nom, cpi).dropna(how="all").dropna(axis=1)
        cpi_floor_label = "CPI Floor (real)"
        floor_series = pd.Series(1.0, index=rets.index, name=cpi_floor_label)
    else:
        rets = rets_nom.copy()
        # If nominal, plot the CPI index as a rising floor line (normalize to 1)
        cpi_idx = (cpi["CPI"] / cpi["CPI"].iloc[0]).reindex(rets.index)
        floor_series = cpi_idx.rename("CPI Level (normalized)")

    # Align columns after any dropped series
    tickers = [t for t in tickers if t in rets.columns]

    # Risk‑free monthly
    rf_m = (1.0 + rf_ann)**(1.0/12.0) - 1.0

    # ---------- Optimization ----------
    w = optimize(rets, method, rf_m, enforce_dd, dd_limit)

    # Portfolio series
    port = portfolio_series(w, rets)
    wealth_port = cumulative_wealth(port).rename("Optimized Portfolio")

    # Comparator wealth
    wealth_rf = riskfree_wealth(wealth_port.index, rf_ann)

    # ---------- Charts ----------
    tab1, tab2 = st.tabs(["Wealth Curves", "Weights & Metrics"])

    with tab1:
        st.subheader("Wealth vs CPI floor and Risk‑Free")
        fig, ax = plt.subplots(figsize=(10, 5))

        # Base lines
        if plot_real:
            # CPI floor is a perfectly flat line at 1.0 in real terms
            ax.plot(floor_series.index, floor_series.values, label="CPI Floor", linestyle="--")
        else:
            ax.plot(floor_series.index, floor_series.values, label="CPI (normalized)", linestyle="--")

        ax.plot(wealth_port.index, wealth_port.values, label="Optimized Portfolio", linewidth=2.0)
        ax.plot(wealth_rf.index, wealth_rf.values, label="Risk‑Free (smooth)", linewidth=1.5)

        # Optionally show each asset’s wealth (faint)
        show_assets = st.checkbox("Overlay individual assets (wealth)", value=False)
        if show_assets:
            wealth_assets = (1.0 + rets).cumprod()
            for col in wealth_assets.columns:
                ax.plot(wealth_assets.index, wealth_assets[col].values, alpha=0.35, linewidth=0.9, label=col)

        ax.set_ylabel("Wealth (× initial)")
        ax.set_xlabel("Date")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)

        st.pyplot(fig)

    with tab2:
        left, right = st.columns([0.55, 0.45])
        with left:
            st.subheader("Weights")
            w_df = pd.DataFrame({"Weight": w}, index=rets.columns)
            w_df = w_df.sort_values("Weight", ascending=False)
            st.dataframe((w_df * 100).round(2))

            # Pie / alt visualization
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            nonzero = w_df[w_df["Weight"] > 1e-6]
            ax2.pie(nonzero["Weight"], labels=nonzero.index, autopct="%1.1f%%", startangle=90)
            ax2.axis("equal")
            st.pyplot(fig2)

        with right:
            st.subheader("Portfolio Metrics")
            # Compute key metrics
            wealth = wealth_port
            cagr = cagr_from_wealth(wealth)
            mdd = abs(min(0.0, max_drawdown(port)))  # positive
            mu_a, sig_a = annualize_mean_std(port)
            shrp = sharpe_ratio(w, rets, rf_m)
            srtn = sortino_ratio(w, rets, rf_m)
            clmr = calmar_ratio(w, rets)
            strl = sterling_ratio(w, rets)
            cvar = cvar_loss(w, rets, alpha=0.05)

            met = pd.DataFrame({
                "Value": [
                    f"{cagr:.2%}",
                    f"{mdd:.2%}",
                    f"{mu_a:.2%}",
                    f"{sig_a:.2%}",
                    f"{shrp:.2f}",
                    f"{srtn:.2f}",
                    f"{strl:.2f}",
                    f"{clmr:.2f}",
                    f"{cvar:.2%}"
                ]
            }, index=[
                "CAGR",
                "Max Drawdown",
                "Ann. Return",
                "Ann. Volatility",
                "Sharpe",
                "Sortino",
                "Sterling (approx)",
                "Calmar",
                "CVaR 5% (monthly)"
            ])
            st.table(met)

    st.divider()
    st.caption(
        "Notes: ‘Optimize’ sets the objective for the historical backtest and solves a long‑only weight vector (sum=1). "
        "If **Max Drawdown** is ON, the optimizer enforces the specified drawdown limit over the full sample. "
        "‘CPI floor’ means we plot in real terms so CPI is flat; if you switch off real terms, the CPI line becomes the normalized CPI level. "
        "Risk‑free is a smooth comparator at the chosen annual rate and is not included in the investable set."
    )

else:
    st.info("Pick at least two assets in the sidebar to begin.")
# Safe wrapper: allows `python main.py` without double-starting Streamlit
if __name__ == "__main__":
    import sys
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        # Older Streamlit: no ctx available
        def get_script_run_ctx():
            return None

    # Only invoke the Streamlit CLI if we're NOT already in a Streamlit context
    if get_script_run_ctx() is None:
        try:
            from streamlit.web import cli as stcli
        except Exception:
            from streamlit import cli as stcli
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
