from collections import namedtuple, defaultdict
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
import matplotlib.cm as cm

from bart.common.Utils import area_under_curve
from trappy.utils import handle_duplicate_index

# TODO strip imports

class WaResultsCollector(object):
    def __init__(self, wa_dirs, platform=None):
        self.results_df = pd.DataFrame()
        self.platform = platform

        for wa_dir in wa_dirs:
            df = self.read_wa_dir(wa_dir)
            self.results_df = self.results_df.append(df)

    def read_wa_dir(self, wa_dir):
        # return
        # kernel,id,workload,tag,workload_id,iteration,metric,value,units
        df = pd.read_csv(os.path.join(wa_dir, 'results.csv'))

        with open(os.path.join(wa_dir, '__meta', 'jobs.json')) as f:
            jobs = json.load(f)['jobs']

        subdirs_done = []

        # Keep track of how many times we've seen each job id so we know which
        # iteration to look at (If we use the proper WA3 API this awkwardness
        # isn't necessary).
        next_iteration = defaultdict(lambda: 1)

        # See comment below about splitting job_id
        df['tag'] = df.id.apply(lambda id: id.split('-')[0])
        df['workload_id'] = df.id.apply(lambda id: id.split('-')[1])

        for job in jobs:
            workload = job['workload_name']

            job_id = job['id']

            # The 'tag' should describe the userspace configuration or other
            # changes made to the system without changing the kernel code under
            # test. Right now we are using the 'id' field of each 'section'
            # entry for that, but this is not really right.
            #
            # The workload_id identifies it among the 'workloads' entry in the
            # agenda - basically it identifies a workload+workload_parameters
            # tuple.
            #
            # Right now we are identifying both of them by splitting up the
            # job's id, but this is not correct. We should be using
            # 'classifiers' to do that, but I don't know how yet.
            tag, workload_id = job_id.split('-')

            iteration = next_iteration[job_id]
            next_iteration[job_id] += 1

            job_dir = os.path.join(wa_dir,
                                   '-'.join([job_id, workload, str(iteration)]))

            extra_df = self.get_extra_job_metrics(job_dir, workload)

            if extra_df is not None:
                extra_df.loc[:, 'workload'] = workload
                extra_df.loc[:, 'iteration'] = iteration
                extra_df.loc[:, 'id'] = job_id
                extra_df.loc[:, 'tag'] = tag
                extra_df.loc[:, 'workload_id'] = workload_id

                df = df.append(extra_df)

        df.loc[:, 'kernel'] = self.get_kernel_version(wa_dir)

        return df

    def get_extra_job_metrics(self, job_dir, workload):
        # return
        # value,metric,units
        if workload == 'jankbench':
            df = pd.read_csv(os.path.join(job_dir, 'jankbench_frames.csv'))
            df = pd.DataFrame({'value': df['total_duration']})
            df.loc[:, 'metric'] = 'frame_total_duration'
            df.loc[:, 'units'] = 'ms'

            return df

        return None

    def get_kernel_version(self, wa_dir):
        with open(os.path.join(wa_dir, '__meta', 'target_info.json')) as f:
            target_info = json.load(f)
        return KernelVersion(target_info['kernel_release']).sha1

    def _select(self, tag='.*', kernel='.*', workload_id='.*'):
        _df = self.results_df
        _df = _df[_df.tag.str.match(tag)]
        _df = _df[_df.kernel.str.match(kernel)]
        _df = _df[_df.workload_id.str.match(workload_id)]
        return _df

    def plot_total_duration(self, tag='.*', kernel='.*', workload_id='.*',
                            by=['workload_id', 'tag', 'kernel'], tmax=32):
        workload = 'jankbench'
        metric = 'frame_total_duration'

        df = (self._select(tag, kernel, workload_id)
              .groupby(['workload', 'metric'])
              .get_group((workload, metric)))

        units = df['units'].unique()
        if len(units) > 1:
            raise RuntimError('Found different units for workload "{}" metric "{}": {}'
                              .format(workload, metric, units))
        [units] = units

        # Sort groups by mean duration - this will be the order of the plots
        gb = df.groupby(by)

        # Convert the groupby into a DataFrame with a column for each group
        max_group_size = max(len(group) for group in gb.groups.itervalues())
        _df = pd.DataFrame()
        for group_name, group in gb:
            # Need to pad the group's column so that they all have the same
            # length
            padding_length = max_group_size - len(group)
            padding = pd.Series(np.nan, index=np.arange(padding_length))
            col = group['value'].append(padding)
            col.index = np.arange(max_group_size)
            _df[group_name] = col

        # Sort the columns so that the groups with the lowest mean get plotted
        # at the top
        avgs = _df.mean()
        avgs = avgs.sort_values(ascending=False)
        _df = _df[avgs.index]

        # Plot boxes sorted by mean
        fig, axes = plt.subplots(figsize=(16,8))
        _df.boxplot(ax=axes, vert=False, showmeans=True)
        fig.suptitle('')
        axes.set_xlim(0,tmax)
        axes.set_xlabel('[{}]'.format(units))
        plt.show()

    CDF = namedtuple('CDF', ['df', 'threshold', 'above', 'below'])

    def get_cdf(self, data, threshold):
        """
        Build the "Cumulative Distribution Function" (CDF) for the given data
        """
        # Build the series of sorted values
        ser = data.sort_values()
        if len(ser) < 1000:
            # Append again the last (and largest) value.
            # This step is important especially for small sample sizes
            # in order to get an unbiased CDF
            ser = ser.append(pd.Series(ser.iloc[-1]))
        df = pd.Series(np.linspace(0., 1., len(ser)), index=ser)

        # Compute percentage of samples above/below the specified threshold
        below = float(max(df[:threshold]))
        above = 1 - below
        return self.CDF(df, threshold, above, below)

    def plot_cdf(self, workload='jankbench', metric='frame_total_duration',
                 threshold=16, tag='.*', kernel='.*', workload_id='.*'):

        df = (self._select(tag, kernel, workload_id)
              .groupby(['workload', 'metric'])
              .get_group((workload, metric)))

        units = df['units'].unique()
        if len(units) > 1:
            raise RuntimError('Found different units for workload "{}" metric "{}": {}'
                              .format(workload, metric, units))
        [units] = units

        test_cnt = len(df.groupby(['workload_id', 'tag', 'kernel']))
        colors = iter(cm.rainbow(np.linspace(0, 1, test_cnt+1)))

        fig, axes = plt.subplots()
        axes.axvspan(0, threshold, facecolor='g', alpha=0.1);

        labels = []
        lines = []
        for keys, df in df.groupby(['workload_id', 'tag', 'kernel']):
            labels.append("{:16s}: {:32s}".format(keys[2], keys[1]))
            color = next(colors)
            cdf = self.get_cdf(df['value'], threshold)
            ax = cdf.df.plot(ax=axes, legend=False, xlim=(0,None), figsize=(16, 6),
                             title='Total duration CDF ({:.1f}% within {} [{}] threshold)'\
                             .format(100. * cdf.below, threshold, units),
                             label=workload_id, color=color)
            lines.append(ax.lines[-1])
            axes.axhline(y=cdf.below, linewidth=1,
                         linestyle='--', color=color)
            print("%-32s: %-32s: %.1f", keys[2], keys[1], 100.*cdf.below)

        axes.grid(True)
        axes.legend(lines, labels)
        plt.show()
