import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def analyze_performance(df):
    df_mean = df.groupby("id_conf").mean(numeric_only=True)
    # Calcul de l'efficacité : HV produit par seconde de calcul
    df_mean['efficiency'] = df_mean['hv'] / df_mean['duration_batch']
    return df_mean


def save_analysis_plots(df, top_ids, output_name="analysis_results.png"):
    df_filtered = df[df['id_conf'].isin(top_ids)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    sns.boxplot(data=df_filtered, x='id_conf', y='hv', ax=ax1, palette="viridis")
    ax1.set_title('Distribution de l\'Hypervolume (Top HV)')
    ax1.tick_params(axis='x', rotation=45)

    sns.boxplot(data=df_filtered, x='id_conf', y='spacing', ax=ax2, palette="magma")
    ax2.set_title('Distribution du Spacing (Top HV)')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_name)
    plt.close()


if __name__ == "__main__":
    # Configuration des chemins
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
    Dir_gridsearch = "results_parameters"
    cvs_file = ROOT_DIR / Dir_gridsearch / "grid_search.csv"

    data = load_data(cvs_file)
    stats = analyze_performance(data)

    # 1. Top 15 par Hypervolume (Performance pure)
    print("--- TOP 15 PAR HYPERVOLUME ---")
    print(stats.nlargest(15, "hv")[['hv', 'spacing', 'duration_batch', 'efficiency']])

    # 2. Top 15 par Efficacité (Meilleur ratio Performance/Temps)
    print("\n--- TOP 15 PAR EFFICACITÉ (Uniquement si HV > 0.60) ---")
    stats_filtered = stats[stats['hv'] > 0.60]
    top_efficiency = stats_filtered.nlargest(15, "efficiency").sort_values('hv')
    print(top_efficiency)

    # Génération des graphiques pour les 10 meilleurs HV
    top_ids_for_plot = stats.nlargest(10, "hv").index.tolist()
    save_analysis_plots(data, top_ids_for_plot)

    print(f"\nGraphiques de distribution sauvegardés sous: analysis_results.png")