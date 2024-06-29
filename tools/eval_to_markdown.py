from argparse import ArgumentParser
import os

import pandas as pd

DETAILED_CSV_FILENAME = 'pedestrian_detailed.csv'
COLUMN_NAMES = {
    'seq': 'Sequence',
    'HOTA___AUC': 'HOTA',
    'DetA___AUC': 'DetA',
    'AssA___AUC': 'AssA',
    'LocA___AUC': 'LocA',
    'MOTA': 'MOTA',
    'MOTP': 'MOTP',
    'IDF1': 'IDF1',
}
SELECT_COLUMNS = ['Sequence', 'HOTA', 'DetA', 'AssA', 'LocA', 'MOTA', 'MOTP', 'IDF1']

def to_markdown(tracker_dir: str):
    """Convert the TrackEval results to a markdown table ready to be copied.

    Args:
        tracker_dir (str): Path to the directory containing the evaluation results.
    """
    eval_csv = os.path.join(tracker_dir, DETAILED_CSV_FILENAME)

    df = pd.read_csv(eval_csv)
    df = df.rename(columns=COLUMN_NAMES)
    df = df[SELECT_COLUMNS]
    df[df.select_dtypes(include=['number']).columns] *= 100 # Convert to %

    print(df.to_markdown(index=False, floatfmt='3.1f'))


if __name__ == '__main__':
    parser = ArgumentParser(prog='Converts evaluation results into a markdown table.')
    parser.add_argument('tracker_dir', type=str, help='Path to the evaluation directory of the tracker.')

    args = parser.parse_args()

    to_markdown(args.tracker_dir)

# python eval_to_markdown.py ../data/trackers/rdtrack-val/run4
