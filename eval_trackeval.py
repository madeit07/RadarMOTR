from packaging import version
from argparse import Namespace, ArgumentParser
import contextlib
import os

import numpy as np

import TrackEval.trackeval as trackeval

def run_trackeval(tracker_dir: str, args: Namespace):
    """Run TrackEval to get HOTA, MOTA and IDF1 metrics.

    Args:
        tracker_dir (str): Path to the directory containing the track predictions.
        args (Namespace): User arguments
    """

    # np.float, ... are deprecated but still used in TrackEval
    if version.parse(np.__version__) >= version.parse("1.20.0"):
        np.float = np.float32
        np.int = np.int32
        np.bool = bool

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config['DISPLAY_LESS_PROGRESS'] = False
    eval_config['USE_PARALLEL'] = True
    eval_config['NUM_PARALLEL_CORES'] = args.workers
    eval_config['PLOT_CURVES'] = False

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config['BENCHMARK'] = args.dataset
    dataset_config['GT_FOLDER'] = args.data_path
    dataset_config['SPLIT_TO_EVAL'] = args.split
    dataset_config['SKIP_SPLIT_FOL'] = False
    dataset_config['DO_PREPROC'] = False
    dataset_config['TRACKERS_TO_EVAL'] = [args.tracker_dirname]
    dataset_config['TRACKER_SUB_FOLDER'] = ''
    dataset_config['TRACKERS_FOLDER'] = args.output_dir

    metrics_config = {
        'METRICS': ['HOTA', 'CLEAR', 'Identity'],
        'THRESHOLD': 0.5
    }

    # Temporary write print statements to file
    with open(os.path.join(tracker_dir, 'eval.log'), 'w') as f:
        with contextlib.redirect_stdout(f):
            evaluator = trackeval.Evaluator(eval_config)
            dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
            metrics_list = []
            for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
                if metric.get_name() in metrics_config['METRICS']:
                    metrics_list.append(metric(metrics_config))

            evaluator.evaluate(dataset_list, metrics_list)

if __name__ == '__main__':
    parser = ArgumentParser(description='Runs TrackEval to evaluate trackers performance based on HOTA, CLEAR and Identity metrics.')
    parser.add_argument('--version', '-v', action='version', version='1.0.0')

    parser.add_argument('--data', '-d', dest='data_path', type=str, required=True, help='Path to the directory containing the splits of the radar sequences.')
    parser.add_argument('--dataset', '-ds', dest='dataset', type=str, required=True, help='Name of the dataset. Default is directory name of data path.')
    parser.add_argument('--tracker', '-t', dest='tracker_dirname', type=str, required=True, help='Path to the evaluation directory of the tracker.')
    parser.add_argument('--output', '-o', dest='output_dir', type=str, default=os.path.join('data', 'trackers'), help='Path to the directory where all trackers are stored grouped by the dataset split.')
    parser.add_argument('--split', '-s', dest='split', type=str, default='val', help='Dataset split to evaluate.')
    parser.add_argument('--workers', '-j', dest='workers', type=int, default=2, help='Number of workers for parallel execution.')

    args = parser.parse_args()

    args.dataset = args.dataset.lower()
    if args.workers < 1:
        raise ValueError('Worker argument must be a positive non-zero integer.')

    output_dir = os.path.join(args.output_dir, f'{args.dataset}-{args.split}', args.tracker_dirname)
    os.makedirs(output_dir, exist_ok=True)
    print(f'Output directory: \"{output_dir}\".')

    run_trackeval(output_dir, args)

    print(f'Written results to: \"{os.path.join(output_dir, "eval.log")}\"')
