import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})

DISPLAY_NAME = {
    "dns_time_ms": "DNS Time (ms)",
    "tcp_connect_ms": "TCP Connect Time (ms)",
    "tls_time_ms": "TLS Handshake Time (ms)",
    "rtt_ms": "Round-Trip Time (ms)",
    "response_size": "Response Size (bytes)",
    "response_time_ms": "Response Time (ms)",
    "time_of_day": "Time of Day",
    "time_sin": "Time of Day (sin)",
    "time_cos": "Time of Day (cos)",
    "domain_category": "Domain Category",
    "baseline_mean": "Baseline (Mean)",
    "linear_regression": "Linear Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
}


def nice(name):
    """Human-readable label, falls back to title-cased underscores-to-spaces."""
    return DISPLAY_NAME.get(name, name.replace("_", " ").title())

DATA_PATH = Path(__file__).resolve().parent / "Dataset_finale.csv"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------- 1. LOAD
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    print(f"[load] shape={df.shape}")
    return df


# ---------------------------------------------------------------- 2. CLEAN (structural, pre-split)
def clean_structural(df):
    df = df.copy()
    n_null, n_dup = df.isnull().sum().sum(), df.duplicated().sum()
    print(f"[clean] nulls={n_null}, dupes={n_dup}")
    if n_null:
        df = df.dropna()
    if n_dup:
        df = df.drop_duplicates()

    before = len(df)
    df = df[df["http_status"] == 200].copy()
    print(f"[clean] dropped {before - len(df)} non-200 rows")
    df = df.drop(columns=["http_status"])

    before = len(df)
    comp_sum = df["dns_time_ms"] + df["tcp_connect_ms"] + df["tls_time_ms"]
    df = df[comp_sum <= df["response_time_ms"]].copy()
    print(f"[clean] dropped {before - len(df)} cross-field-broken rows")

    df["domain_category"] = df["domain_category"].str.strip().str.lower()
    print(f"[clean] structural clean done, shape={df.shape}")
    return df


def split_data(df, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    print(f"[split] train={train_df.shape}, test={test_df.shape}")
    return train_df, test_df


def clean_statistical(train_df, test_df, numeric_cols):
    """Fit outlier rule on TRAIN only, apply identical rule to both."""
    model = IsolationForest(contamination=0.02, random_state=42)
    model.fit(train_df[numeric_cols])

    train_pred = model.predict(train_df[numeric_cols])
    test_pred = model.predict(test_df[numeric_cols])

    train_clean = train_df[train_pred == 1].copy()
    test_clean = test_df[test_pred == 1].copy()

    print(f"[clean-stat] train: dropped {len(train_df)-len(train_clean)} anomalies "
          f"({len(train_clean)} remain)")
    print(f"[clean-stat] test: dropped {len(test_df)-len(test_clean)} anomalies "
          f"({len(test_clean)} remain)")
    return train_clean, test_clean


# ---------------------------------------------------------------- 3. ANALYZE
def analyze(df, label="train"):
    print(f"\n[analyze:{label}] describe:")
    print(df.describe())
    print(f"\n[analyze:{label}] domain_category counts:")
    print(df["domain_category"].value_counts())


# ---------------------------------------------------------------- 4. FEATURE ENGINEERING
def encode_time_cyclical(df):
    td = df["time_of_day"]
    h, m, s = td // 10000, (td // 100) % 100, td % 100
    bad = (h > 23) | (m > 59) | (s > 59)
    if bad.any():
        raise ValueError(f"invalid HHMMSS values: {td[bad].tolist()}")
    total_seconds = h * 3600 + m * 60 + s
    theta = 2 * np.pi * (total_seconds / 86400)
    df["time_sin"] = np.sin(theta)
    df["time_cos"] = np.cos(theta)
    return df.drop(columns=["time_of_day"])


def feature_engineer(train_df, test_df):
    train_df = encode_time_cyclical(train_df.copy())
    test_df = encode_time_cyclical(test_df.copy())

    # one-hot encode domain_category, fit categories on train, align test
    train_df = pd.get_dummies(train_df, columns=["domain_category"], drop_first=True)
    test_df = pd.get_dummies(test_df, columns=["domain_category"], drop_first=True)
    test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

    print(f"[feat] train cols: {list(train_df.columns)}")
    return train_df, test_df


def scale_features(train_df, test_df, feature_cols):
    """Fit scaler on train only, transform both."""
    scaler = StandardScaler()
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_scaled, test_scaled, scaler


# ---------------------------------------------------------------- 5. BASELINE / MODEL
def train_models(X_train, y_train):
    models = {}

    class MeanBaseline:
        def fit(self, X, y):
            self.value = y.mean()
            return self
        def predict(self, X):
            return np.full(len(X), self.value)

    baseline = MeanBaseline().fit(X_train, y_train)
    models["baseline_mean"] = baseline

    lr = LinearRegression().fit(X_train, y_train)
    models["linear_regression"] = lr

    dt = DecisionTreeRegressor(max_depth=6, random_state=42).fit(X_train, y_train)
    models["decision_tree"] = dt

    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42).fit(X_train, y_train)
    models["random_forest"] = rf

    print(f"[model] trained: {list(models.keys())}")
    return models


# ---------------------------------------------------------------- 6. EVALUATE
def evaluate(models, X_test, y_test):
    rows = []
    preds = {}
    for name, m in models.items():
        pred = m.predict(X_test)
        preds[name] = pred
        mape = np.mean(np.abs((y_test.values - pred) / y_test.values)) * 100
        rows.append({
            "model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred),
            "MAPE_%": mape,
            "Accuracy_%": 100 - mape,
        })
    result = pd.DataFrame(rows)
    print("\n[evaluate]")
    print(result.to_string(index=False))
    return result, preds


# ---------------------------------------------------------------- 7. VISUALIZE
def visualize_feature_target(df, target="response_time_ms"):
    """Feature-vs-target scatter plots. Property of the data, not any
    one model's output - generated once, independent of models."""
    feat_dir = OUT_DIR / "feature_vs_target"
    feat_dir.mkdir(exist_ok=True)

    numeric_feats = [c for c in df.select_dtypes(include="number").columns
                      if c != target]

    for feat in numeric_feats:
        plt.figure(figsize=(6, 4))
        plt.scatter(df[feat], df[target], alpha=0.4, color="#4C72B0",
                    label="data point (one request)")
        plt.xlabel(nice(feat))
        plt.ylabel(nice(target))
        plt.title(f"{nice(feat)} vs {nice(target)}")
        plt.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.savefig(feat_dir / f"{feat}_vs_target.png", dpi=120)
        plt.close()

    print(f"[visualize] saved {len(numeric_feats)} feature-vs-target plots -> {feat_dir}")


def visualize(y_test, preds, results, model_names=None):
    if model_names is None:
        model_names = list(preds.keys())

    for model_name in model_names:
        pred = preds[model_name]
        residual = y_test.values - pred
        row = results[results["model"] == model_name].iloc[0]
        metrics_text = (f"MAE = {row['MAE']:.1f}\nRMSE = {row['RMSE']:.1f}\n"
                         f"R² = {row['R2']:.3f}\nAccuracy = {row['Accuracy_%']:.1f}%")
        title_model = nice(model_name)

        model_dir = OUT_DIR / model_name
        model_dir.mkdir(exist_ok=True)

        # 1. actual vs predicted
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, pred, alpha=0.5, color="#4C72B0",
                    label="test request (actual vs. predicted)")
        lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
        plt.plot(lims, lims, color="#C44E52", linestyle="--",
                  label="perfect prediction line")
        plt.xlabel("Actual Response Time (ms)")
        plt.ylabel("Predicted Response Time (ms)")
        plt.title(f"Actual vs Predicted — {title_model}")
        plt.legend(loc="upper left", fontsize=8, frameon=False)
        plt.gca().text(0.98, 0.02, metrics_text, transform=plt.gca().transAxes,
                        fontsize=9, va="bottom", ha="right",
                        bbox=dict(boxstyle="round", facecolor="white",
                                  edgecolor="#cccccc", alpha=0.9))
        plt.tight_layout()
        plt.savefig(model_dir / "actual_vs_predicted.png", dpi=120)
        plt.close()

        # 2. residual plot
        plt.figure(figsize=(6, 4))
        plt.scatter(pred, residual, alpha=0.5, color="#4C72B0",
                    label="test request (error at that prediction)")
        plt.axhline(0, color="#C44E52", linestyle="--", label="zero error")
        plt.xlabel("Predicted Response Time (ms)")
        plt.ylabel("Residual: Actual − Predicted (ms)")
        plt.title(f"Residual Plot — {title_model}")
        plt.legend(loc="upper right", fontsize=8, frameon=False)
        plt.tight_layout()
        plt.savefig(model_dir / "residual_plot.png", dpi=120)
        plt.close()

        # 3. residual histogram
        plt.figure(figsize=(6, 4))
        plt.hist(residual, bins=30, color="#4C72B0",
                  label="number of test requests")
        plt.axvline(0, color="#C44E52", linestyle="--", label="zero error")
        plt.xlabel("Residual: Actual − Predicted (ms)")
        plt.ylabel("Number of Test Requests")
        plt.title(f"Residual Distribution — {title_model}")
        plt.legend(loc="upper right", fontsize=8, frameon=False)
        plt.tight_layout()
        plt.savefig(model_dir / "residual_hist.png", dpi=120)
        plt.close()

        print(f"[visualize] saved plots for {model_name} -> {model_dir}")


# ---------------------------------------------------------------- MAIN
if __name__ == "__main__":
    df = load_data()
    df = clean_structural(df)
    train_df, test_df = split_data(df)

    numeric_cols = ["dns_time_ms", "tcp_connect_ms", "tls_time_ms", "rtt_ms",
                     "response_size", "response_time_ms"]
    train_df, test_df = clean_statistical(train_df, test_df, numeric_cols)

    analyze(train_df, "train")
    visualize_feature_target(train_df)

    train_df, test_df = feature_engineer(train_df, test_df)

    feature_cols = [c for c in train_df.columns if c != "response_time_ms"]
    train_df, test_df, scaler = scale_features(train_df, test_df, feature_cols)

    X_train, y_train = train_df[feature_cols], train_df["response_time_ms"]
    X_test, y_test = test_df[feature_cols], test_df["response_time_ms"]

    models = train_models(X_train, y_train)
    results, preds = evaluate(models, X_test, y_test)
    visualize(y_test, preds, results)

    results.to_csv(OUT_DIR / "model_results.csv", index=False)
    print(f"\n[main] pipeline complete. results saved to {OUT_DIR/'model_results.csv'}")