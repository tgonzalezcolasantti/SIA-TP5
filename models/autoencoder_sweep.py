from itertools import product
import multiprocessing
from multiprocessing.pool import ApplyResult, ThreadPool
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd
from rich.live import Live
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskID, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from data.parser import dumb_parse_c_array
from models.activation import Activation, Logistic, ReLU, Tanh
from models.autoencoder import Autoencoder
from models.optimization import Adam, NoOptimization, Optimization


topologies = [
    [35,8,2],
    [35,16,2],
    [35,16,8,2]
]
etas = [0.01, 0.005, 0.001]
optimizations = [Adam, NoOptimization]
activations = [Tanh(), ReLU(), Logistic()]
epochs_list = [2000, 10000, 50000]
seeds = [42]

class AutoencoderConfig(TypedDict):
    topology: List[int]
    eta: float
    optimization: type[Optimization]
    activation: Activation
    epochs: int
    seed: int
    metric: Optional[float]

def run_task(data: np.ndarray, config: AutoencoderConfig, task: TaskID, progress: Progress) -> Tuple[Autoencoder, AutoencoderConfig]:
    model = Autoencoder(
        train_inputs=data,
        topology=config['topology'],
        eta=config['eta'],
        encoder_activation=config['activation'],
        decoder_activation=config['activation'],
        epochs=config['epochs'],
        batch_size=(len(data)),
        optimization=config['optimization'],
        rng=np.random.default_rng(config['seed']),
        task=task,
        progress=progress
    )
    reconstructed = model.reconstruct(data) >= 0.5
    config['metric'] = np.mean(np.count_nonzero(reconstructed != data, axis=1))
    return model, config

def run_study(max_tasks: Optional[int] = None):
    inputs, labels = dumb_parse_c_array(Path(__file__).parent.resolve() / '..' / 'data' / 'font.h')

    taskprogress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=True,
    )
    globalprogress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    table = Table(box=None)
    table.add_row(taskprogress)
    table.add_row(globalprogress)

    rows: List[AutoencoderConfig] = []
    settings = list(product(topologies, etas, optimizations, activations, epochs_list, seeds))
    with (
        ThreadPool(processes=max_tasks or int(multiprocessing.cpu_count())) as executor,
        Live(table, refresh_per_second=10),
    ):
        jobs: List[ApplyResult] = []
        for config_id, (topo, eta, opti, acti, epoch, seed) in enumerate(settings):
            task = taskprogress.add_task(
                f"topo={topo} eta={eta} opt={str(opti())[0:2]} acti={str(acti)[0:2]} epochs={epoch} ({seed})",
                start=False,
                total=epoch,
                visible=False,
                is_task=True,
            )
            config: AutoencoderConfig = {"topology": topo, "eta": eta, "optimization": opti, "activation": acti, "epochs": epoch, "seed": seed, "metric": None}
            jobs.append(
                executor.apply_async(
                    run_task, (inputs, config, task, taskprogress)
                )
            )
        full_progress = globalprogress.add_task(
            f"Total progress ({len(jobs)} elements)", total=len(jobs), is_task=False
        )
        while len(jobs) > 0:
            for job in list(jobs):
                if job.ready():
                    jobs.remove(job)
                    row: Tuple[Autoencoder, AutoencoderConfig] = job.get()
                    if row is None:
                        raise ValueError("OOPS")
                    globalprogress.advance(full_progress)
                    row[1]['optimization'] = row[1]['optimization']() # type: ignore
                    for key in row[1].keys():
                        row[1][key] = str(row[1][key])
                    rows.append(row[1])
                    # print(f"Configuración {row['config_id']}/{len(settings)} terminada", flush=True)
        trials = pd.DataFrame(rows)
        trials.to_csv(Path(__file__).parent.resolve() / '..' / "autoencoder_sweep.csv", index=False)

if __name__ == '__main__':
    run_study()