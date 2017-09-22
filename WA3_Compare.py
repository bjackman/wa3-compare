
from collections import namedtuple
import csv
import json
import numpy as np
import pandas as pd
import os
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from subprocess import check_output
from devlib.target import KernelVersion
from trace import Trace

from bart.common.Utils import area_under_curve
from trappy.utils import handle_duplicate_index

def get_additional_metrics(trace_path, platform=None):
    events = ['irq_handler_entry', 'cpu_frequency', 'sched_load_cfs_rq', 'nohz_kick']
    trace = Trace(platform, trace_path, events)

    yield 'cpu_wakeup_count', len(trace.data_frame.cpu_wakeups()), None

    clusters = trace.platform.get('clusters')
    if clusters:
        for cluster in clusters.values():
            name = '-'.join(str(c) for c in cluster)

            df = trace.data_frame.cluster_frequency_residency(cluster)
            df = df.reset_index()
            avg_freq = (df.frequency * df.time).sum() / df.time.sum()
            metric = 'avg_freq_cluster_{}'.format(name)
            yield metric, avg_freq, 'MHz'

            df = trace.data_frame.trace_event('cpu_frequency')
            df = df[df.cpu == cluster[0]]
            yield 'freq_transition_count_{}'.format(name), len(df), None

    if trace.hasEvents('sched_load_cfs_rq'):
        df = trace.data_frame.trace_event('sched_load_cfs_rq')
        util_sum = (handle_duplicate_index(df[lambda r: r.path == '/'])
                    .pivot(columns='cpu').util.ffill().sum(axis=1))
        avg_util_sum = area_under_curve(util_sum) / (util_sum.index[-1] - util_sum.index[0])
        yield 'avg_util_sum', avg_util_sum, None

    if trace.hasEvents('nohz_kick'):
        yield 'nohz_kick_count', len(trace.data_frame.trace_event('nohz_kick')), None

def get_kernel_version(wa_dir):
    with open(os.path.join(wa_dir, '__meta', 'target_info.json')) as f:
        target_info = json.load(f)
    return KernelVersion(target_info['kernel_release']).sha1


def get_results_summary(results_path):
    cached_csv_path = os.path.join(results_path, 'lisa_results.csv')
    if os.path.exists(cached_csv_path):
        return pd.read_csv(cached_csv_path)

    csv_path = os.path.join(results_path, 'results.csv')
    df = pd.read_csv(csv_path)

    WlSpec = namedtuple('WlSpec', ['name', 'iterations'])

    with open(os.path.join(results_path, '__meta', 'jobs.json')) as f:
        jobs = json.load(f)['jobs']

    subdirs_done = []

    i = 0

    for job in jobs:
        workload = job['workload_name']
        id = job['id']
        section = id.split('-')[0] # TODO WA3 sould expose this
        for iteration in range(1, job['iterations'] + 1):
            subdir = '-'.join([id, workload, str(iteration)])

            if subdir in subdirs_done:
                # TODO: I think this is a bug: for 5 global iterations we get 5
                # jobs, I think there should only be one?
                continue

            print 'parsing trace {}'.format(i)
            i += 1

            trace_path = os.path.join(results_path, subdir, 'trace.dat')
            for metric, value, units in get_additional_metrics(trace_path):
                df = df.append(pd.DataFrame({
                    'workload': workload, 'id': id, 'iteration': iteration,
                    'metric': metric, 'value': value, 'units': units,
                    'trace_path': trace_path
                }, index=[df.index[-1] + 1]))

            subdirs_done.append(subdir)

    kver = get_kernel_version(results_path)
    df['kernel'] = pd.Series(kver for _ in range(len(df))) # um

    df['section'] = df.id.apply(lambda id: id.split('-')[0])
    df.to_csv(cached_csv_path)

    return df

def get_results(*dirs):
    return reduce(lambda df1, df2: df1.append(df2), map(get_results_summary, dirs))

def compare_dirs(base_id, results_df, by='kernel'):
    comparisons = []
    df = results_df

    Comparison = namedtuple('Comparison', ['metric', 'wl_id',
                                           'base_id', 'base_mean', 'base_std',
                                           'new_id', 'new_mean', 'new_std',
                                           'diff', 'diff_pct', 'pvalue'])

    # If comparing by kernel, only check comparisons where the 'section' is the same
    # If comparing by section, only check where kernel is same
    if by == 'kernel':
        invariant = 'section'
    elif by == 'section':
        invariant = 'kernel'
    else:
        raise ValueError('`by` must be "kernel" or "section"')

    for metric, group in df.groupby('metric'):
        print 'comparing {}'.format(metric)

        for (workload, inv_id), wl_conf_group in group.groupby(['workload', invariant]):
            gb = wl_conf_group.groupby(by)['value']

            print 'comparing {} {}'.format(workload, inv_id)

            if base_id not in gb.groups:
                print 'Skipping - No baseline results for workload [{}] {} [{}] metric [{}]'.format(
                    workload, invariant, inv_id, metric)
                continue

            base_df = gb.get_group(base_id)
            base_mean = base_df.mean()

            for group_id, df in gb:

                if group_id == base_id:
                    continue

                new_mean = df.mean()
                mean_diff = new_mean - base_mean
                mean_diff_pct = mean_diff * 100. / base_mean
                pvalue =  ttest_ind(df, base_df, equal_var=False).pvalue
                comparisons.append(Comparison(
                    metric, '_'.join([workload, str(inv_id)]),
                    base_id, base_mean, base_df.std(),
                    group_id, new_mean, df.std(),
                    mean_diff, mean_diff_pct, pvalue))

    return pd.DataFrame(comparisons)

# sha_map = {}
# source_path = '/home/brendan/sources/linux/'

# df = comparisons

# for sha in df.new_id.unique().tolist() + df.base_id.unique().tolist():
#     tag = check_output(['git', '-C', source_path, 'describe', '--all', sha]).strip()
#     sha_map[sha] = tag

# for sha, tag in sha_map.iteritems():
#     df = df.replace(sha, tag)


def drop_unchanged_metrics(df):
    # Drop metrics where none of the kernels showed a statistically significant
    # difference
    ignored_metrics = []
    for metric, mdf in df.groupby('metric'):
        if not any(mdf.pvalue < 0.1):
            print 'min-p={:05.2f} diff_pct={:04.1f}  - Ignoring metric [{}] '.format(
                mdf.pvalue.min(), mdf.diff_pct.mean(), metric)
            ignored_metrics.append(metric)
    df = df[~df.metric.isin(ignored_metrics)]
    return df

def plot_comparisons(df):
    for wl_id, workload_comparisons in df.groupby('wl_id'):
        fig, ax = plt.subplots(figsize=(15, len(workload_comparisons) / 2.))

        thickness=0.3
        pos = np.arange(len(workload_comparisons['metric'].unique()))
        colors = ['r', 'g', 'b']
        for i, (group, gdf) in enumerate(workload_comparisons.groupby('new_id')):

            bars = ax.barh(bottom=pos + (i * thickness), width=gdf['diff_pct'],
                           height=thickness, label=group,
                           color=colors[i % len(colors)], align='center')
            for bar, pvalue in zip(bars, gdf['pvalue']):
                bar.set_alpha(1 - (min(pvalue * 10, 0.95)))

        # add some text for labels, title and axes ticks
        ax.set_xlabel('Percent difference')
        [baseline] = workload_comparisons['base_id'].unique()
        ax.set_title('{}: Percent difference compared to {} \nopacity depicts p-value'.format(wl_id, baseline))
        ax.set_yticklabels(gdf['metric'])
        ax.set_yticks(pos + thickness / 2)
        ax.legend(loc='best')

        ax.grid(True)
        # ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
