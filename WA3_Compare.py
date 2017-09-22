
# coding: utf-8

# In[1]:

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')


# In[2]:

from collections import defaultdict, namedtuple
from itertools import combinations
import csv
import json
import numpy as np
import pandas as pd
import os
import sys
from scipy.stats import ttest_ind
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
from subprocess import check_output
from devlib.target import KernelVersion
from trace import Trace
from conf import JsonConf


# In[3]:

# from conf import LisaLogging
# import logging
# LisaLogging.setup(level=logging.DEBUG)


# In[4]:

platform = JsonConf('/home/brendan/sources/lisa/libs/utils/platforms/hikey960.json').load()


# In[5]:

from bart.common.Utils import area_under_curve
from trappy.utils import handle_duplicate_index


# In[6]:

def get_additional_metrics(trace_path):
    trace = Trace(platform, trace_path, ['irq_handler_entry', 'cpu_frequency', 'sched_load_cfs_rq', 'nohz_kick'])

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
        util_sum = (handle_duplicate_index(trace.data_frame.trace_event('sched_load_cfs_rq')[lambda r: r.path == '/'])
                    .pivot(columns='cpu').util.ffill().sum(axis=1))
        avg_util_sum = area_under_curve(util_sum) / (util_sum.index[-1] - util_sum.index[0])
        yield 'avg_util_sum', avg_util_sum, None

    if trace.hasEvents('nohz_kick'):
        yield 'nohz_kick_count', len(trace.data_frame.trace_event('nohz_kick')), None


# In[7]:

def get_kernel_version(wa_dir):
    with open(os.path.join(wa_dir, '__meta', 'target_info.json')) as f:
        target_info = json.load(f)
    return KernelVersion(target_info['kernel_release']).sha1


# In[13]:

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
                # TODO: I think this is a bug: for 5 global iterations we get 5 jobs, I think there should only be one?
                continue

            print 'parsing trace {}'.format(i)
            i += 1

            trace_path = os.path.join(results_path, subdir, 'trace.dat')
            for metric, value, units in get_additional_metrics(trace_path):
                df = df.append(pd.DataFrame({
                    'workload': workload, 'id': id, 'section': section, 'iteration': iteration,
                    'metric': metric, 'value': value, 'units': units,
                    'trace_path': trace_path
                }, index=[df.index[-1] + 1]))

            subdirs_done.append(subdir)

    kver = get_kernel_version(results_path)
    df['kernel'] = pd.Series(kver for _ in range(len(df))) # um

    df.to_csv(cached_csv_path)

    return df


# In[14]:

def get_results(*dirs):
    return reduce(lambda df1, df2: df1.append(df2), map(get_results_summary, dirs))

def compare_dirs(base_kernel, results_df, by='kernel'):
    comparisons = []
    df = results_df

    Comparison = namedtuple('Comparison', ['metric', 'wl_id',
                                           'base_id', 'base_mean', 'base_std',
                                           'new_id', 'new_mean', 'new_std',
                                           'diff', 'diff_pct', 'pvalue'])

    for metric, group in df.groupby('metric'):
        print 'comparing {}'.format(metric)
        
        for (workload, conf_id), wl_conf_group in group.groupby(['workload', 'id']):
            gb = wl_conf_group.groupby('kernel')['value']
            
            if base_kernel not in gb.groups:
                print 'Skipping - No baseline results for workload [{}] id [{}] metric [{}]'.format(
                    workload, conf_id, metric)
                continue
                
            base_df = gb.get_group(base_kernel)
            base_mean = base_df.mean()

            for kernel, df in gb:
                if kernel == base_kernel:
                    continue

                new_mean = df.mean()
                mean_diff = new_mean - base_mean
                mean_diff_pct = mean_diff * 100. / base_mean
                pvalue =  ttest_ind(df, base_df, equal_var=False).pvalue
                comparisons.append(Comparison(
                    metric, '_'.join([workload, str(conf_id)]),
                    base_kernel, base_mean, base_df.std(), 
                    kernel, new_mean, df.std(),
                    mean_diff, mean_diff_pct, pvalue))

    return pd.DataFrame(comparisons)

# In[15]:

dirs = [
    '/home/brendan/sources/wa3-stuff/wa_output.hikey960_orig+waltfix/'
#     '/home/brendan/sources/wa-stuff/wa_outputs-testing/nohz-updates-2/baseline',
#     '/home/brendan/sources/wa-stuff/wa_outputs-testing/nohz-updates-2/vingu-original',
#     '/home/brendan/sources/wa-stuff/wa_outputs-testing/nohz-updates-2/tick-instead',
#     '/home/brendan/sources/wa-stuff/wa_outputs-testing/nohz-updates-2/full-stack',
]

results = get_results(*dirs)
base_kernel = get_kernel_version(dirs[0])
comparisons = compare_dirs(base_kernel, results)


# In[16]:

comparisons


# In[ ]:

sha_map = {}
source_path = '/home/brendan/sources/linux/'

df = comparisons

for sha in df.new_id.unique().tolist() + df.base_id.unique().tolist():
    tag = check_output(['git', '-C', source_path, 'describe', '--all', sha]).strip()
    sha_map[sha] = tag

for sha, tag in sha_map.iteritems():
    df = df.replace(sha, tag)


# In[ ]:

sha_map


# In[ ]:

# Drop metrics where none of the kernels showed a statistically significant difference
ignored_metrics = []
for metric, mdf in df.groupby('metric'):
    if not any(mdf.pvalue < 0.1):
        print 'min-p={:05.2f} diff_pct={:04.1f}  - Ignoring metric [{}] '.format(mdf.pvalue.min(), mdf.diff_pct.mean(), metric)
        ignored_metrics.append(metric)
df = df[~df.metric.isin(ignored_metrics)]


# In[ ]:

fig, ax = plt.subplots(figsize=(15, len(df) / 2.))

thickness=0.3
pos = np.arange(len(df['metric'].unique()))
colors = ['r', 'g', 'b']
for i, (kernel, kdf) in enumerate(df.groupby('new_id')):
    bars = ax.barh(bottom=pos + (i * thickness), width=kdf['diff_pct'], height=thickness, label=kernel,
                color=colors[i % len(colors)], align='center')
    for bar, pvalue in zip(bars, kdf['pvalue']):
        bar.set_alpha(1 - (min(pvalue * 10, 0.95)))

# add some text for labels, title and axes ticks
ax.set_xlabel('Percent difference')
[baseline] = df['base_id'].unique()
ax.set_title('Percent difference compared to {} \nopacity depicts p-value'.format(baseline))
ax.set_yticklabels(df['metric'].unique())
ax.set_yticks(pos + thickness / 2)
ax.legend(loc='best')

ax.grid(True)
# ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))


# In[ ]:

EVENTS=['sched_switch', 'irq_handler_entry', 'cpu_frequency', 'sched_load_cfs_rq']


# In[ ]:

from trappy.plotter import plot_trace


# In[ ]:

from trappy.utils import handle_duplicate_index


# In[ ]:

def get_traces(kernel):
    paths = results[results.kernel == kernel].trace_path.dropna().unique()
    return [Trace(platform, path, EVENTS) for path in paths]


# In[ ]:

base_traces = get_traces('c59bcd9')
vingu_traces = get_traces('cee7b97')


# In[ ]:

trace1 = base_traces[0]
trace2 = vingu_traces[0]


# In[ ]:

import matplotlib.pyplot as plt


# In[ ]:

def get_cum_wakeups(traces):
    df = pd.DataFrame()
    for i, trace in enumerate(traces):
        _df = trace.data_frame.cpu_wakeups()
        _df = pd.DataFrame(np.arange(len(_df)), index=_df.index, columns=[i])
        df = df.join(_df, how='outer')
    return df.ffill()
fig, ax = plt.subplots(figsize=(20, 9))
get_cum_wakeups(base_traces).plot(ax=ax, color='r')
get_cum_wakeups(vingu_traces).plot(ax=ax, color='b')
ax.legend().set_visible(False)


# In[ ]:

def do_plots(trace):
    trace.analysis.frequency.plotClusterFrequencies()
#     trace.analysis.cpus.plotCPU()
    df = (handle_duplicate_index(trace.data_frame.trace_event('sched_load_cfs_rq'))
          .pivot(columns='cpu')[lambda r: r.path == '/'].util)
    df.plot(figsize=(20, 3))

    df = trace.data_frame.cpu_wakeups()
    df.reset_index().plot(kind='scatter', x='Time', y='__cpu', figsize=(20, 3))

    df = pd.DataFrame(np.arange(len(df)), index=df.index)
    df.plot(figsize=(20, 3))

do_plots(trace1)
do_plots(trace2)


# In[ ]:

p = trace1.data_dir + '/trace.dat'
get_ipython().system(u'kernelshark $p')


# In[ ]:

get_ipython().system(u'trace-cmd report $p | grep cpu_idle | head -100')


# In[ ]:

pd.DataFrame([0, 1, 2]).tolist()


# In[ ]:

def count_wakeups(trace):
    return trace.data_frame.cpu_wakeups().groupby('__cpu')['name'].describe()['count']


# In[ ]:

def count_freq_transitions(trace):
    return trace.data_frame.trace_event('cpu_frequency').groupby('cpu')['frequency'].describe()['count']


# In[ ]:

df1 = count_wakeups(trace1)
df2 = count_wakeups(trace2)


# In[ ]:

df2 - df1 


# In[ ]:

df1 = count_freq_transitions(trace1)
df2 = count_freq_transitions(trace2)


# In[ ]:

df2 - df1


# In[ ]:

get_ipython().system(u'trace-cmd report $p | grep irq_handler_entry | head')


# In[ ]:

trace1.analysis.frequency.plotCPUFrequencies(cpus=[0, 1])


# In[ ]:

trace2.analysis.frequency.plotCPUFrequencies(cpus=[0, 1])

