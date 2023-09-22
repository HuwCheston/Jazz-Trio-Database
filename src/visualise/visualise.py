from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import utils
import src.visualise.visualise_utils as vutils


class ScatterPlotByBeat(vutils.BasePlot):
    def __init__(self, item, **kwargs):
        self.item = item
        super().__init__(figure_title=rf'onsets_plots\scatterplot_bybeat_{self.item.item["fname"]}', **kwargs)
        self.df = pd.DataFrame(self.format_df())
        self.cmap = sns.color_palette(
            kwargs.get('cmap', 'husl'),
            n_colors=len(utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys()),
            as_cmap=False
        )
        self.fig, self.ax = plt.subplots(
            nrows=1,
            ncols=len(utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys()),
            sharex=True,
            sharey=True,
            figsize=(vutils.WIDTH, 7)
        )

    def format_df(self):
        for instr in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys():
            instr_ons = self.item.ons[instr]
            z = zip(self.item.ons['mix'], self.item.ons['mix'][1:], self.item.ons['metre_manual'])
            for beat1, beat2, beat1pos in z:
                vals = instr_ons[np.logical_and(beat1 <= instr_ons, instr_ons < beat2)]
                for i in vals:
                    pos = ((i - beat1) / (beat2 - beat1)) + beat1pos
                    yield {
                        'instrument': instr,
                        'timestamp': pd.to_datetime(datetime.fromtimestamp(beat1).strftime('%H:%M:%S')),
                        'musical_position': pos if pos < 4.9 else pos - 4
                    }

    def _create_plot(self):
        for ax, (idx, grp), col in zip(self.ax.flatten(), self.df.groupby('instrument'), self.cmap):
            g = sns.scatterplot(data=grp, x='musical_position', y='timestamp', ax=ax, color=col, s=40, legend=None)
            g.set_title(idx.title())

    def _format_ax(self):
        minor_ticks = [i + f for i in range(1, 5) for f in (1 / 3, 2 / 3)]
        for ax in self.ax.flatten():
            ax.set(xlim=(0.8, 5.2), xticks=list(range(1, self.item['time_signature'] + 1)), xlabel='', ylabel='')
            ax.set_xticks(minor_ticks, labels=[], minor=True)
            ax.grid(which='major', ls='-', lw=1)
            ax.grid(which='minor', ls='--', lw=0.3)
            plt.setp(ax.spines.values(), linewidth=2)
            ax.tick_params(axis='both', width=2)

    def _format_fig(self):
        self.fig.supxlabel(r'Beat ($\frac{1}{4}$ note)')
        self.fig.supylabel('Time (s)')
        self.fig.suptitle(self.item.item['fname'])
        self.fig.subplots_adjust(left=0.11, right=0.95, top=0.85, bottom=0.15)


#
# plt.rcParams.update({'font.size': 18})
# def f(row):
#     if int(row['year']) < 1963:
#         val = 'Trio 1 (1959 – 1962)\nLeFaro/Motian'
#     elif int(row['year']) < 1966:
#         val = 'Trio 2 (1963 – 1965)\nIsraels/Bunker'
#     elif int(row['year']) < 1975:
#         val = 'Trio 3 (1968 – 1974)\nGomez/Morell'
#     else:
#         val = 'Trio 4 (1978 – 1980)\nJohnson/LeBarbera'
#     return val
#
# big = pd.concat(dfs).reset_index(drop=True)
# big['trio'] = big.apply(f, axis=1)
# melt = big.melt(id_vars=['year', 'instrument', 'performer', 'trio', 'n_observations'], value_vars=['coupling_bass', 'coupling_piano', 'coupling_drums'])
# melt = melt[melt['n_observations'] >= 30].sort_values(by=['year', 'instrument'])
#
# fig, ax = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(20, 6.5))
# for a, (idx, grp) in zip(ax.flatten(), melt.groupby('trio')):
#     grp = grp.sort_values(by=['instrument', 'performer'])
#     g = sns.barplot(
#         data=grp, x='instrument', y='value', hue='variable', hue_order=['coupling_bass', 'coupling_drums', 'coupling_piano'], ax=a,
#         estimator=np.nanmean, errorbar='ci', n_boot=1000, errcolor='black', errwidth=2, edgecolor='black', lw=2, capsize=0.1, width=0.8, palette='tab10'
#     )
#     g.get_legend().remove()
#     g = sns.stripplot(data=grp, x='instrument', y='value', hue='variable', hue_order=['coupling_bass', 'coupling_drums', 'coupling_piano'], ax=a, dodge=True, legend=False, color='black', s=4)
#     g.set(title=idx, ylabel='', xlabel='', xticklabels=['Bass', 'Drums', 'Piano'])
#     g.axhline(y=0, xmin=0, xmax=3, color='black', lw=2)
#     plt.setp(a.spines.values(), linewidth=2)
#     a.tick_params(axis='both', width=2)
#
# fig.legend(hand, ['Bass', 'Drums', 'Piano',], loc='right', title='Influencer', frameon=False)
# fig.supxlabel('Influenced')
# fig.supylabel('Coupling coefficient')
# fig.subplots_adjust(left=0.075, right=0.9, top=0.85, bottom=0.15)
# plt.show()




# # Plotting parameter optimization results sketches
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# import src.utils.analyse_utils as autils
#
# plt.rcParams.update({'font.size': 18})
#
# csvs = []
# for instr in autils.INSTRS_TO_PERF.keys():
#     csvs.append(pd.DataFrame(autils.load_csv(rf'{autils.get_project_root()}\references\parameter_optimisation', f'onset_detect_{instr}')))
# csvs.append(pd.DataFrame(autils.load_csv(rf'{autils.get_project_root()}\references\parameter_optimisation', f'beat_track_mix')))
# df = pd.concat(csvs).reset_index(drop=True)
#
# for idx, grp in df.groupby(['instrument', 'iterations']):
#     assert len(grp) == 21
#     assert grp['fname'].duplicated().any() == False
#
# df['instrument'] = df['instrument'].str.title()
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18.8, 5), sharex=False, sharey=True)
# g = sns.lineplot(
#     data=df, x='iterations', y='f_score', hue='instrument', errorbar='sd', lw=3,
#     estimator=np.mean, err_kws=dict(alpha=0.2), n_boot=1000, seed=1, ax=ax[0],
#     hue_order=['Piano', 'Bass', 'Drums', 'Mix']
# )
# g.set(xlabel='Optimization iterations', ylabel='Objective function ($F$)', xlim=(-5, 500))
# g.axhline(y=1, xmin=g.get_xlim()[0], xmax=g.get_xlim()[1], color='black', alpha=0.3, ls='--', lw=3)
# sns.move_legend(g, loc='lower right', title='Instrument')
# ax[0].tick_params(width=3, )
# plt.setp(ax[0].spines.values(), linewidth=2)
#
#
# mima = df.groupby(['instrument', 'mbz_id']).agg({'f_score': ['min', 'max']})
# mima.columns = mima.columns.droplevel()
# opt = df[df.groupby(['instrument'])['iterations'].transform(max) == df['iterations']].groupby(['instrument', 'mbz_id'])['f_score'].mean()
# combined = pd.concat([mima, opt], axis=1).rename(columns=dict(min='Worst run', max='Best run', f_score='Optimized')).reset_index(drop=False).melt(id_vars=['instrument', 'mbz_id'], var_name='Parameters')
#
# g = sns.barplot(
#     data=combined, x='Parameters', y='value', hue='instrument', estimator=np.mean,
#     errorbar=('ci', 95), ax=ax[1], lw=2, width=0.8, errcolor='black', errwidth=3,
#     n_boot=1000, seed=1, capsize=0.075, edgecolor='black', hue_order=['Piano', 'Bass', 'Drums', 'Mix']
# )
# g = sns.stripplot(
#     data=combined, x='Parameters', y='value', hue='instrument', ax=ax[1], alpha=0.5,
#     dodge=True, palette='dark:black',legend=False, s=6, marker='.', jitter=0.1, hue_order=['Piano', 'Bass', 'Drums', 'Mix']
# )
# g.set(xlabel='', ylabel='')
# sns.move_legend(g, loc='best', title='Instrument')
# ax[1].tick_params(width=3, )
# plt.setp(ax[1].spines.values(), linewidth=2)
# fig.suptitle('Optimizing Bill Evans corpus ($N$=21) with NLopt "Sbplx" algorithm')
# fig.subplots_adjust(top=0.85, bottom=0.15, right=0.975, left=0.05, wspace=0.1)
#
# plt.show()