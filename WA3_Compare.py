
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

# TODOs:
# - There are some things I want to change about the WA3 output format
# - This uses workload names where it should use the job id (but it also needs
#   to compare across sections, which are also included in the job ID)
# - Need to think about what happens if the worklod Id changes between Runs

def get_cpu_time(trace, cpus):
    df = pd.DataFrame([trace.getCPUActiveSignal(cpu) for cpu in cpus])
    return df.sum(axis=1).sum(axis=0)

def get_trace_metrics(trace_path, platform=None):
    metrics = []

    events = ['irq_handler_entry', 'cpu_frequency', 'nohz_kick', 'sched_switch',
              'sched_load_cfs_rq', 'sched_load_avg_task']
    trace = Trace(platform, trace_path, events)

    metrics.append(('cpu_wakeup_count', len(trace.data_frame.cpu_wakeups()), None))

    clusters = trace.platform.get('clusters')
    if clusters:
        for cluster in clusters.values():
            name = '-'.join(str(c) for c in cluster)

            df = trace.data_frame.cluster_frequency_residency(cluster)
            if df is None or df.empty:
                print "Can't get cluster freq residency from {}".format(trace.data_dir)
            else:
                df = df.reset_index()
                avg_freq = (df.frequency * df.time).sum() / df.time.sum()
                metric = 'avg_freq_cluster_{}'.format(name)
                metrics.append((metric, avg_freq, 'MHz'))

            df = trace.data_frame.trace_event('cpu_frequency')
            df = df[df.cpu == cluster[0]]
            metrics.append(('freq_transition_count_{}'.format(name), len(df), None))

            active_time = area_under_curve(trace.getClusterActiveSignal(cluster))
            metrics.append(('active_time_cluster_{}'.format(name), active_time, 'seconds'))

            metrics.append(('cpu_time_cluster_{}'.format(name), get_cpu_time(trace, cluster), 'cpu-seconds'))

    metrics.append(('cpu_time_total', get_cpu_time(trace, range(trace.platform['cpus_count'])), 'cpu-seconds'))

    event = None
    if trace.hasEvents('sched_load_cfs_rq'):
        event = 'sched_load_cfs_rq'
        row_filter = lambda r: r.path == '/'
        column = 'util'
    elif trace.hasEvents('sched_load_avg_cpu'):
        event = 'sched_load_avg_cpu'
        row_filter = lambda r: True
        column = 'util_avg'
    if event:
        df = trace.data_frame.trace_event(event)
        util_sum = (handle_duplicate_index(df)[row_filter]
                    .pivot(columns='cpu')[column].ffill().sum(axis=1))
        avg_util_sum = area_under_curve(util_sum) / (util_sum.index[-1] - util_sum.index[0])
        metrics.append(('avg_util_sum', avg_util_sum, None))

    if trace.hasEvents('nohz_kick'):
        metrics.append(('nohz_kick_count', len(trace.data_frame.trace_event('nohz_kick')), None))

    return [{'metric': m, 'value': v, 'units': u, 'trace_path': trace_path}
            for m, v, u in metrics]

def get_additional_metrics(results_path, id, workload, iteration, platform=None):
    subdir = '-'.join([id, workload, str(iteration)])

    metrics = []

    trace_path = os.path.join(results_path, subdir, 'trace.dat')
    if os.path.exists(trace_path):
        metrics += get_trace_metrics(trace_path)

    if workload == 'jankbench':
        with open(os.path.join(results_path, subdir, 'jankbench_frames.csv')) as f:
            reader = csv.DictReader(f)
            for row in reader:
                metrics.append({'metric': 'frame_total_duration',
                                'value': float(row['total_duration']),
                                'units': 'ms'})

    return pd.DataFrame(metrics)

def get_kernel_version(wa_dir):
    with open(os.path.join(wa_dir, '__meta', 'target_info.json')) as f:
        target_info = json.load(f)
    return KernelVersion(target_info['kernel_release']).sha1


def get_results_summary(results_path, platform=None):
    cached_csv_path = os.path.join(results_path, 'lisa_results.csv')
    # if os.path.exists(cached_csv_path):
    #     return pd.read_csv(cached_csv_path)

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

            # TODO rework this so that the additional stuff returns DataFrames
            trace_path = os.path.join(results_path, subdir, 'trace.dat')
            extra_df = get_additional_metrics(
                results_path, id, workload, iteration, platform=platform)
            extra_df.loc[:, 'workload'] = workload
            extra_df.loc[:, 'iteration'] = iteration
            extra_df.loc[:, 'trace_path'] = trace_path
            extra_df.loc[:, 'id'] = id

            df = df.append(extra_df)

            subdirs_done.append(subdir)

    kver = get_kernel_version(results_path)
    df.loc[:, 'kernel'] = kver

    print df['id'].unique()

    # TODO: this is wrong and bad
    # Need to stop relying on 'section'. Hopefully using WA API can make this
    # better
    try:
        df['section'] = df.id.apply(lambda id: id.split('-')[0])
        df['wl_id'] = df.id.apply(lambda id: id.split('-')[1])
    except IndexError:
        # Probably no sections
        df.loc[:, 'section'] = None
        df['wl_id'] = df['id']

    df.to_csv(cached_csv_path)

    return df

def get_results(dirs, platform=None):
    dfs = [get_results_summary(d, platform=platform) for d in dirs]
    return reduce(lambda df1, df2: df1.append(df2), dfs)

def compare_dirs(base_id, results_df, by='kernel'):
    """
    Take a DataFrame of WA results (from get_results) and find metrics that changed

    This takes a DataFrame as returned by get_results, and returns a DataFrame
    showing the changes in metrics, with respect to a given baseline, for all
    variants it finds. The notion of 'variant' and 'baseline' is defined by the
    `by` param. If by='kernel', then `base_id` should be a kernel SHA (or
    whatever key the 'kernel' column in the results_df uses). If by='section'
    then `base_id` should be a WA 'section id' (as named in the WA agenda).
    """
    comparisons = []

    # I dunno why I wrote this with a namedtuple instead of just a dict or
    # whatever, but it works nicely
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

    if base_id not in results_df[by].unique():
        raise ValueError('base_id "{}" not a valid "{}" (available: {}). '
                         'Did you mean to set by="{}"?'.format(
                             base_id, by, results_df[by].unique().tolist(), invariant))

    for metric, metric_results in results_df.groupby('metric'):
        # inv_id will either be the id of the kernel or of the section,
        # depending on the `by` param.
        # So wl_inv_results will be the results entries for that workload on
        # that kernel/section
        for (workload, inv_id), wl_inv_results in metric_results.groupby(['wl_id', invariant]):
            gb = wl_inv_results.groupby(by)['value']

            if base_id not in gb.groups:
                print 'Skipping - No baseline results for workload [{}] {} [{}] metric [{}]'.format(
                    workload, invariant, inv_id, metric)
                continue

            base_results = gb.get_group(base_id)
            base_mean = base_results.mean()

            for group_id, group_results in gb:
                if group_id == base_id:
                    continue

                # group_id is now a kernel id or a section id (depending on
                # `by`). group_results is a slice of all the rows of results_df
                # for a given metric, workload, section/workload tuple. We
                # create comparison object to show how that metric changed
                # wrt. to the base section/workload.

                group_mean = group_results.mean()
                mean_diff = group_mean - base_mean
                mean_diff_pct = mean_diff * 100. / base_mean
                pvalue =  ttest_ind(group_results, base_results, equal_var=False).pvalue
                comparisons.append(Comparison(
                    metric, '_'.join([workload, str(inv_id)]),
                    base_id, base_mean, base_results.std(),
                    group_id, group_mean, group_results.std(),
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
        # ax.set_xlim((-50, 50))
        ax.legend(loc='best')

        ax.grid(True)
        # ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
