import argparse
import pandas as pd
from rl_utils.plotting.auto_table import plot_table

if __name__ == '__main__':
    df = pd.DataFrame([
        {
            'method': 'gala',
            'perf': 90.0,
            'err': 2.0,
            'rank': '0'
        }
    ])

    plot_table(df, 'method', 'rank', 'perf', col_order=['gala'], row_order=['0'], err_key='err')

