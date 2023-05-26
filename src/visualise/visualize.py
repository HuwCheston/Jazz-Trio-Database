# import seaborn as sns
# import matplotlib.pyplot as plt
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
