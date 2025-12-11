#!/usr/bin/env python
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import pdfcombine

def select(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    new_columns = [each for each in df.columns if each.startswith(prefix)]
    new_columns = ['epoch', 'step'] + new_columns

    df_new = df[new_columns]

    df_new.columns = [each.removeprefix(prefix)
                      for each in df_new.columns]
    df_new = df_new[np.logical_not(np.isnan(df_new.loss))]
    return df_new # type: ignore


def get_unique_epoch(df):
    epoch = df['epoch'].to_numpy()[:-1]
    _, index = np.unique(np.flip(epoch), return_index=True)
    index = len(epoch) - index - 1
    return df[['step', 'epoch']].iloc[index]


def plot_learning_curve(df_train: pd.DataFrame,
         df_val: pd.DataFrame,
         df_test: pd.DataFrame,
         df_epoch: pd.DataFrame,
         y: str,
         x: str = 'step',
):
    fig, ax = plt.subplots(figsize=(10, 10))

    if y in df_train.columns:
        ax.plot(df_train[x], df_train[y], label='Training', color='tab:blue', alpha=0.5)
        train_smooth_x, train_smooth_y = lowess(endog=df_train[y], exog=df_train[x], frac=0.075, it=0, is_sorted=True).T
        ax.plot(train_smooth_x, train_smooth_y, label='Training (LOWESS)', color='tab:blue', lw=3)
    ax.plot(df_val[x], df_val[y], label='Validation', color='tab:orange', lw=3)
    ax.plot(df_test[x], df_test[y], label='Test', color='tab:red', ls='', marker='*', markersize=20)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(y)


    xticks = df_epoch['step']
    xticklabels = df_epoch['epoch']
    if len(xticklabels) > 10:
        xtick_step = len(df_epoch) // 5
        xticks = xticks[::xtick_step]
        xticklabels = xticklabels[::xtick_step]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    ax.legend()
    ax.grid()
    return fig


def make_learning_curves(log_dir: Path):
    ckpt_path = next(log_dir.glob('checkpoints/**/best*.ckpt'))
    best_info = dict(each.split('=') for each in ckpt_path.stem.split('-') if '=' in each)
    best_info = {key: int(value) for key, value in best_info.items()}

    metrics_path = log_dir / 'metrics.csv'
    df = pd.read_csv(metrics_path)

    df_train = select(df, 'train_')
    df_val = select(df, 'val_')
    df_test = select(df, 'test_')

    df_test['epoch'] = best_info['epoch']
    df_test['step'] = best_info['step']

    df_epoch = get_unique_epoch(df)

    output_dir = log_dir / 'learning-curve'
    output_dir.mkdir()

    df_train.to_csv(output_dir / 'training.csv.xz')
    df_val.to_csv(output_dir / 'validation.csv')
    df_test.to_csv(output_dir / 'test.csv')
    df_epoch.to_csv(output_dir / 'epoch.csv')


    pdf_list: list[str] = []

    for metric in df_val.columns:
        if metric in ['step', 'epoch']:
            continue
        fig = plot_learning_curve(df_train=df_train, df_val=df_val, df_test=df_test, df_epoch=df_epoch, y=metric)
        output_path = output_dir / metric
        for suffix in ['.pdf', '.png']:
            fig.savefig(output_path.with_suffix(suffix))
        plt.close(fig)

        pdf_list.append(str(output_path.with_suffix('.pdf')))

    combined_pdf = log_dir / 'learning-curve.pdf'
    pdfcombine.combine(files=pdf_list, output=str(combined_pdf))
