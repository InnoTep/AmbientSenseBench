import os
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_PATH = "data/results/wp6_anomaly_detection_results.csv"
FIGURE_DIR = "data/results/figures"

os.makedirs(FIGURE_DIR, exist_ok=True)


def load_results():
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(
            f"Missing results file: {RESULTS_PATH}. "
            "Run src/run_wp6_anomaly_detection.py first."
        )

    df = pd.read_csv(RESULTS_PATH)

    required_columns = [
        "scenario",
        "model",
        "auprc",
        "f1_fixed_threshold",
        "n_true_alarms",
        "n_false_alarms",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in WP6 results file: {missing}")

    return df


def shorten_model_name(model_name):
    if model_name == "Isolation Forest":
        return "IF"
    if model_name == "One-Class SVM":
        return "OCSVM"
    return model_name


def plot_grouped_bar(df, value_col, ylabel, title, output_name, y_min=None, y_max=None):
    """
    Draw a grouped bar chart by scenario and model.
    """
    pivot = df.pivot(index="scenario", columns="model", values=value_col)
    pivot = pivot[["Isolation Forest", "One-Class SVM"]]

    ax = pivot.plot(kind="bar", figsize=(8, 5))

    ax.set_xlabel("Scenario")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Model")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)

    plt.xticks(rotation=0)
    plt.tight_layout()

    output_path = os.path.join(FIGURE_DIR, output_name)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_alarm_composition(df):
    """
    Plot true and false alarm composition as horizontal stacked bars.
    """
    plot_df = df.copy()

    model_order = {
        "Isolation Forest": 0,
        "One-Class SVM": 1,
    }

    plot_df["model_order"] = plot_df["model"].map(model_order)
    plot_df = plot_df.sort_values(["scenario", "model_order"]).reset_index(drop=True)

    plot_df["label"] = (
        plot_df["scenario"] + " " + plot_df["model"].apply(shorten_model_name)
    )

    y_positions = range(len(plot_df))

    plt.figure(figsize=(8, 5))

    plt.barh(
        y_positions,
        plot_df["n_true_alarms"],
        label="True alarms",
    )

    plt.barh(
        y_positions,
        plot_df["n_false_alarms"],
        left=plot_df["n_true_alarms"],
        label="False alarms",
    )

    plt.yticks(y_positions, plot_df["label"])
    plt.xlabel("Number of alarms")
    plt.ylabel("Scenario and model")
    plt.title("WP6 Alarm Composition under Fixed Training-Derived Threshold")
    plt.legend()
    plt.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()

    output_path = os.path.join(FIGURE_DIR, "wp6_alarm_composition.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    df = load_results()

    plot_grouped_bar(
        df=df,
        value_col="auprc",
        ylabel="AUPRC",
        title="WP6 AUPRC by Scenario and Model",
        output_name="wp6_auprc_comparison.png",
        y_min=0.85,
        y_max=1.02,
    )

    plot_grouped_bar(
        df=df,
        value_col="f1_fixed_threshold",
        ylabel="Fixed-threshold F1",
        title="WP6 Fixed-threshold F1 by Scenario and Model",
        output_name="wp6_fixed_f1_comparison.png",
        y_min=0,
        y_max=1.0,
    )

    plot_alarm_composition(df)

    print("\nWP6 selected figures generated.")


if __name__ == "__main__":
    main()