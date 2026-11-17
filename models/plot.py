"""Gráficos desde resultados guardados, siguiendo ej2/plot_comparisons.py del TP3."""

import argparse
import csv
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional

import matplotlib
from matplotlib.axes import Axes

from data.parser import dumb_parse_c_array
from models.activation import ReLU
from models.autoencoder import Autoencoder, MonitoringAutoencoder
from models.optimization import Adam
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_results(results_dir):
    out = Path(results_dir)
    plots = out / "results"
    plots.mkdir(parents=True, exist_ok=True)

    def save_figure(fig, name):
        fig.tight_layout()
        fig.savefig(plots / name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    csv_file = results_dir / 'autoencoder_sweep.csv'
    
    time_vs_triangles_map = parse_csv(csv_file, ['eta'], 'topology', 'metric', {'optimization': "Adam(0.9,0.999)"})
    print(time_vs_triangles_map)
    bar_graph(plots / 'error_vs_topo.png', time_vs_triangles_map, "Pixeles erroneos vs topologia", "Error")

    time_vs_triangles_map = parse_csv(csv_file, ['activation'], 'optimization', 'metric', {"topology": "[35, 16, 2]"})
    print(time_vs_triangles_map)
    bar_graph(plots / 'error_vs_acti.png', time_vs_triangles_map, "Pixeles erroneos vs optimizacion y activacion", "Error")

    time_vs_triangles_map = parse_csv(csv_file, ['eta'], 'epochs', 'metric', {"topology": "[35, 16, 2]", "optimization": "Adam(0.9,0.999)"})
    print(time_vs_triangles_map)
    bar_graph(plots / 'error_vs_epoch.png', time_vs_triangles_map, "Pixeles erroneos vs epochs", "Error", sorter=lambda x: float(x))

    inputs, _ = dumb_parse_c_array(Path(__file__).parent.resolve() / '..' / 'data' / 'font.h')
    data_1 = MonitoringAutoencoder([35, 16, 2], 0.01, inputs, np.random.default_rng(42), ReLU(), ReLU(), 50000, 32, Adam).get_stats()
    data_2 = MonitoringAutoencoder([35, 16, 2], 0.001, inputs, np.random.default_rng(42), ReLU(), ReLU(), 50000, 32, Adam).get_stats()

    fig, axes = plt.subplots(1, 2, figsize=(10,6))
    ax_0: Axes = axes[0]
    ax_1: Axes = axes[1]

    ax_0.plot(list(data_1.keys()), list(data_1.values()))
    ax_0.set_yscale('log')
    ax_0.set_xlabel('Epoch')
    ax_0.set_ylabel('Suma de pixeles incorrectos entre salidas predecidas y esperadas')
    ax_0.set_title("Evolucion del error durante entrenamiento, eta=0.01")
    ax_1.plot(list(data_2.keys()), list(data_2.values()))
    ax_1.set_yscale('log')
    ax_1.set_xlabel('Epoch')
    ax_1.set_ylabel('Suma de pixeles incorrectos entre salidas predecidas y esperadas')
    ax_1.set_title("Evolucion del error durante entrenamiento, eta=0.001")
    fig.tight_layout()
    fig.savefig(results_dir / "progress.png", dpi=150, bbox_inches="tight")

colors = ["lightcoral", "red", "sandybrown", "darkgoldenrod", "yellowgreen", "lightgreen", "chartreuse", "deepskyblue", "royalblue", "mediumblue", "slateblue", "mediumpurple", "hotpink", "lightpink"]

def bar_graph(output_file, data: Mapping[str, Mapping[str, List[float]]], xlabel: str, ylabel: str, sorter: Optional[Callable[[str], float]] = None, log: bool = False):
    bar_width = 1 / (len(data) + 2)
    fig, ax = plt.subplots(figsize =(10, 8))
    groups = list(data[list(data.keys())[0]].keys())

    for idx, x in enumerate(sorted(data.keys(), key=sorter)):
        bars_x = np.arange(len(data[x])) + bar_width * idx
        y_data = [data[x][group] for group in sorted(groups, key=sorter)]
        # if isinstance(y_data[0], List):
        means = [np.mean(i) for i in y_data] # type: ignore
        errors = [np.std(i) for i in y_data] # type: ignore
        b = plt.bar(bars_x, means, width=bar_width, color = colors[idx],
                        edgecolor='grey', label=x, yerr=errors)
        plt.bar_label(b, [f'{i:.4f}' for i in means], padding=5, rotation=90)
        # else:
        #     b = plt.bar(bars_x, y_data, color=colors[idx], width = bar_width,
        #             edgecolor ='grey', label=x)
        #     plt.bar_label(b, y_data, padding=5, rotation=90)

    plt.xlabel(xlabel, fontweight ='bold', fontsize = 15)
    plt.ylabel(ylabel, fontweight ='bold', fontsize = 15)
    plt.xticks([r + 0.5-bar_width for r in range(len(data[list(data.keys())[0]]))], sorted(groups, key=sorter))
    if log:
        ax.set_yscale('log')
    ax.margins(y=0.2)
    plt.legend()
    #plt.show()
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")

def line_graph(output_file: Path, data: Mapping[str, Mapping[str, List[float]]], xlabel: str, ylabel: str):
    _, ax = plt.subplots(figsize =(20, 8))

    for idx, x in enumerate(data.keys()):
        plot_x = [float(i) for _, i in enumerate(sorted(data[x].keys(), key=float))]
        x_order = [idx for idx, _ in enumerate(sorted(data[x].keys(), key=float))]
        print(plot_x)
        print(x_order)
        plot_y = np.mean(list(data[x][str(plot_x[j])] for j in x_order), axis=1)
        plot_yerr = np.std(list(data[x][str(plot_x[j])] for j in x_order), axis=1)
        print(plot_y)
        plt.errorbar(x = plot_x, y = plot_y, yerr=plot_yerr, label=x)

    plt.xlabel(xlabel, fontweight ='bold', fontsize = 15)
    plt.ylabel(ylabel, fontweight ='bold', fontsize = 15)
    # plt.xticks([r + 1 for r in range(len(data))], sorted(data.keys()))
    ax.set_xscale('log')
    ax.margins(y=0.2)
    plt.legend()
    #plt.show()
    plt.savefig(output_file)


def parse_csv(path: Path, x_main: List[str], x_groupby: str | None, y: str, filter_list: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, List[float]]]:
    mapping: Dict[str, Dict[str, List[float]]] = {}

    with open(path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for line in reader:
            if filter_list is not None:
                valid = True
                for condition in filter_list.items():
                    if str(line[condition[0]]) != condition[1]:
                        print(f"skipping (condition {condition[0]} is {line[condition[0]]} and not {condition[1]})")
                        valid = False
                        break
                if not valid:
                    continue
            algokey = " ".join([line[x] for x in x_main])
            if algokey not in mapping:
                mapping[algokey] = {}
            if x_groupby is not None:
                if line[x_groupby] not in mapping[algokey]:
                    mapping[algokey][str(line[x_groupby])] = [] # type: ignore
                try:
                    ans = float(line[y])
                    mapping[algokey][str(line[x_groupby])].append(ans) # type: ignore
                except ValueError:
                    continue
            else:
                if " " not in mapping[algokey]:
                    mapping[algokey][" "] = [] # type: ignore
                mapping[algokey][" "].append(float(line[y])) # type: ignore
    return mapping

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerar gráficos sin entrenar")
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent.resolve() / '..')
    args = parser.parse_args()


    plot_results(args.results_dir)
