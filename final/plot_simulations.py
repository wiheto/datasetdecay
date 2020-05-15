import matplotlib.pyplot as plt
import numpy as np
from plotje import styler
from matplotlib.colors import LogNorm


# Where the data is saved
datadir = '/home/william/sherlock/scratch/data/datasetdecay/sequentialcomparisons/'
# Where figures should be saved
savedir = '/home/william/work/datasetdecay/'

# different parameters
probrange = np.arange(-10, 10)
trrange = np.arange(0, 1, 0.025)
maxcomparisons = 100

# runs, repeats, numbertrue, correlationstr, probability of true at start
uncorr_true = np.zeros([1000, 11, len(trrange), len(probrange)])
uncorr_false = np.zeros([1000, 11, len(trrange), len(probrange)])
bon_true = np.zeros([1000, 11, len(trrange), len(probrange)])
bon_false = np.zeros([1000, 11, len(trrange), len(probrange)])
fdr_true = np.zeros([1000, 11, len(trrange), len(probrange)])
fdr_false = np.zeros([1000, 11, len(trrange), len(probrange)])
pad_true = np.zeros([1000, 11, len(trrange), len(probrange)])
pad_false = np.zeros([1000, 11, len(trrange), len(probrange)])
pas_true = np.zeros([1000, 11, len(trrange), len(probrange)])
pas_false = np.zeros([1000, 11, len(trrange), len(probrange)])
pai_true = np.zeros([1000, 11, len(trrange), len(probrange)])
pai_false = np.zeros([1000, 11, len(trrange), len(probrange)])

uncorr_null = np.zeros([1000, maxcomparisons])
bon_null = np.zeros([1000, maxcomparisons])
fdr_null = np.zeros([1000, maxcomparisons])
pad_null = np.zeros([1000, maxcomparisons])
pas_null = np.zeros([1000, maxcomparisons])
pai_null = np.zeros([1000, maxcomparisons])

for r in range(0, 1000):
    print(r)
    uncorr_true[r] = np.load(datadir + 'uncorr_true_' + str(r) + '.npy')
    uncorr_false[r] = np.load(datadir + 'uncorr_false_' + str(r) + '.npy')
    bon_true[r] = np.load(datadir + 'bon_true_' + str(r) + '.npy')
    bon_false[r] = np.load(datadir + 'bon_false_' + str(r) + '.npy')
    fdr_true[r] = np.load(datadir + 'fdr_true_' + str(r) + '.npy')
    fdr_false[r] = np.load(datadir + 'fdr_false_' + str(r) + '.npy')
    pad_true[r] = np.load(datadir + 'pad_true_' + str(r) + '.npy')
    pad_false[r] = np.load(datadir + 'pad_false_' + str(r) + '.npy')
    pas_true[r] = np.load(datadir + 'pas_true_' + str(r) + '.npy')
    pas_false[r] = np.load(datadir + 'pas_false_' + str(r) + '.npy')
    pai_true[r] = np.load(datadir + 'pai_true_' + str(r) + '.npy')
    pai_false[r] = np.load(datadir + 'pai_false_' + str(r) + '.npy')
    uncorr_null[r] = np.load(datadir + 'uncorr_null_' + str(r) + '.npy')
    bon_null[r] = np.load(datadir + 'bon_null_' + str(r) + '.npy')
    fdr_null[r] = np.load(datadir + 'fdr_null_' + str(r) + '.npy')
    pad_null[r] = np.load(datadir + 'pad_null_' + str(r) + '.npy')
    pas_null[r] = np.load(datadir + 'pas_null_' + str(r) + '.npy')
    pai_null[r] = np.load(datadir + 'pai_null_' + str(r) + '.npy')


def sequential_number_of_false_positives(data):
    comps = np.array([np.cumsum(data[n]) for n in range(data.shape[0])])
    comps[comps > 0] = 1
    fwe = np.mean(comps, axis=0)
    return fwe


def simultaneous_number_of_false_positives(data):
    return np.zeros(len(data)) + np.sum(data > 0)


# Figure 1 is the number of false positives that exist in the data when there is no correlation
# Simulation 1 was when trrange was 0.
nt = 0


fwe_uncorr = sequential_number_of_false_positives(uncorr_null)
pad_uncorr = sequential_number_of_false_positives(pad_null)
pai_uncorr = sequential_number_of_false_positives(pai_null)
pas_uncorr = sequential_number_of_false_positives(pas_null)

fig, ax = plt.subplots(1, 4, figsize=(10, 3), sharey=True, sharex=True)
ax[0].plot(np.arange(1, maxcomparisons+1), fwe_uncorr, color='dimgray')
ax[0].plot(np.arange(1, maxcomparisons+1),
           np.zeros(maxcomparisons)+0.05, ':', color='black')
ax[1].plot(np.arange(1, maxcomparisons+1), pad_uncorr, color='dimgray')
ax[1].plot(np.arange(1, maxcomparisons+1),
           np.zeros(maxcomparisons)+0.05, ':', color='black')
ax[2].plot(np.arange(1, maxcomparisons+1), pas_uncorr, color='dimgray')
ax[2].plot(np.arange(1, maxcomparisons+1),
           np.zeros(maxcomparisons)+0.05, ':', color='black')
ax[3].plot(np.arange(1, maxcomparisons+1), pai_uncorr, color='dimgray')
ax[3].plot(np.arange(1, maxcomparisons+1),
           np.zeros(maxcomparisons)+0.05, ':', color='black')

titles = ['Uncorrected', r'$\alpha$-debt',
          r'$\alpha$-spending', r'$\alpha$-investing']
for i, a in enumerate(ax):
    a.set_xticks([1, 25, 50, 75, 100])
    a.set_yticks([0, 0.25, 0.5, 0.75, 1])
    styler(a, xlabel='Test Number', ylabel='P(at least 1 FP)',
           aspectsquare=True, title=titles[i])

plt.tight_layout()
fig.savefig(savedir + 'simulation1.svg')


def fdr_per_quadrant(data, splitx=0, splity=0.25, xvar=probrange, yvar=trrange):
    q1 = np.mean(data[yvar > splity][:, xvar < splitx])
    q2 = np.mean(data[yvar > splity][:, xvar > splitx])
    q3 = np.mean(data[yvar < splity][:, xvar < splitx])
    q4 = np.mean(data[yvar < splity][:, xvar > splitx])
    return q1, q2, q3, q4


# Take 10 truths in data for most of the simulations
nt = 10
fig, ax = plt.subplots(5, 6, figsize=(12, 10))
ax = ax.flatten()
colorlist = ['steelblue', 'lightsteelblue', 'lightcoral', 'darksalmon']
cmap = plt.cm.inferno
cmap.set_bad(plt.cm.inferno.colors[0])
titles = ['Uncorrected', 'Bonferroni', 'FDR', r'$\alpha$-debt',
          r'$\alpha$-spending', r'$\alpha$-investing']


ax[0].imshow(np.mean(uncorr_true, axis=0)[nt]/nt, origin='lower', extent=[-10,
                                                                          10, 0, 1], aspect='auto', vmin=0, vmax=1, cmap=cmap, rasterized=True)
ax[6].imshow(np.mean(uncorr_false, axis=0)[nt]/(maxcomparisons-nt), origin='lower',  extent=[-10,
                                                                                             10, 0, 1], aspect='auto', vmin=0.00001, vmax=0.1, norm=LogNorm(), cmap=cmap, rasterized=True)
a = np.mean(uncorr_false, axis=0)[
    nt]/(np.mean(uncorr_false, axis=0)[nt]+np.mean(uncorr_true, axis=0)[nt])
ax[12].imshow(a, origin='lower',  extent=[-10, 10, 0, 1], aspect='auto',
              vmin=0.0001, vmax=1, norm=LogNorm(), cmap=cmap, rasterized=True)
q = fdr_per_quadrant(a)
bar = ax[18].bar(['Q1', 'Q2', 'Q3', 'Q4'], q, color='darkgray')
for i, b in enumerate(bar):
    b.set_color(colorlist[i])
ax[18].plot([-0.5, 3.5], [0.05, 0.05], ':', color='black')

q = []
# when nti = 0 is all zeros cause that data was saved as sim1, so start at nti=1
for nti in range(1, 11):
    a = np.mean(uncorr_false, axis=0)[
        nti]/(np.mean(uncorr_false, axis=0)[nti]+np.mean(uncorr_true, axis=0)[nti])
    q.append(fdr_per_quadrant(a))
q = np.array(q)
ax[24].plot(np.arange(1, 11), q[:, 0], color=colorlist[0], alpha=1)
ax[24].plot(np.arange(1, 11), q[:, 1], color=colorlist[1], alpha=1)
ax[24].plot(np.arange(1, 11), q[:, 2], color=colorlist[2], alpha=1)
ax[24].plot(np.arange(1, 11), q[:, 3], color=colorlist[3], alpha=1)
ax[24].plot([1, 10], [0.05, 0.05], ':', color='black')


ax[1].imshow(np.mean(bon_true, axis=0)[nt]/nt, origin='lower',  extent=[-10,
                                                                        10, 0, 1], aspect='auto', vmin=0, vmax=1, cmap=cmap, rasterized=True)
ax[7].imshow(np.mean(bon_false, axis=0)[nt]/(maxcomparisons-nt), origin='lower',  extent=[-10,
                                                                                          10, 0, 1], aspect='auto', vmin=0.00001, vmax=0.1, norm=LogNorm(), cmap=cmap, rasterized=True)
a = np.mean(bon_false, axis=0)[
    nt]/(np.mean(bon_false, axis=0)[nt]+np.mean(bon_true, axis=0)[nt])
q = fdr_per_quadrant(a)
bar = ax[19].bar(['Q1', 'Q2', 'Q3', 'Q4'], q, color='darkgray')
for i, b in enumerate(bar):
    b.set_color(colorlist[i])
ax[19].plot([-0.5, 3.5], [0.05, 0.05], ':', color='black')

q = []
for nti in range(1, 11):
    a = np.mean(bon_false, axis=0)[
        nti]/(np.mean(bon_false, axis=0)[nti]+np.mean(bon_true, axis=0)[nti])
    q.append(fdr_per_quadrant(a))
q = np.array(q)
ax[25].plot(np.arange(1, 11), q[:, 0], color=colorlist[0], alpha=1)
ax[25].plot(np.arange(1, 11), q[:, 1], color=colorlist[1], alpha=1)
ax[25].plot(np.arange(1, 11), q[:, 2], color=colorlist[2], alpha=1)
ax[25].plot(np.arange(1, 11), q[:, 3], color=colorlist[3], alpha=1)
ax[25].plot([1, 10], [0.05, 0.05], ':', color='black')

ax[13].imshow(a, origin='lower',  extent=[-10, 10, 0, 1], aspect='auto',
              vmin=0.0001, vmax=1, norm=LogNorm(), cmap=cmap, rasterized=True)
ax[2].imshow(np.mean(fdr_true, axis=0)[nt]/nt, origin='lower',  extent=[-10,
                                                                        10, 0, 1], aspect='auto', vmin=0, vmax=1, cmap=cmap, rasterized=True)
ax[8].imshow(np.mean(fdr_false, axis=0)[nt]/(maxcomparisons-nt), origin='lower',  extent=[-10,
                                                                                          10, 0, 1], aspect='auto', vmin=0.00001, vmax=0.1, norm=LogNorm(), cmap=cmap, rasterized=True)
a = np.mean(fdr_false, axis=0)[
    nt]/(np.mean(fdr_false, axis=0)[nt]+np.mean(fdr_true, axis=0)[nt])
ax[14].imshow(a, origin='lower',  extent=[-10, 10, 0, 1], aspect='auto',
              vmin=0.0001, vmax=1, norm=LogNorm(), cmap=cmap, rasterized=True)
q = fdr_per_quadrant(a)
bar = ax[20].bar(['Q1', 'Q2', 'Q3', 'Q4'], q, color='darkgray')
for i, b in enumerate(bar):
    b.set_color(colorlist[i])
ax[20].plot([-0.5, 3.5], [0.05, 0.05], ':', color='black')

q = []
for nti in range(1, 11):
    a = np.mean(fdr_false, axis=0)[
        nti]/(np.mean(fdr_false, axis=0)[nti]+np.mean(fdr_true, axis=0)[nti])
    q.append(fdr_per_quadrant(a))
q = np.array(q)
ax[26].plot(np.arange(1, 11), q[:, 0], color=colorlist[0], alpha=1)
ax[26].plot(np.arange(1, 11), q[:, 1], color=colorlist[1], alpha=1)
ax[26].plot(np.arange(1, 11), q[:, 2], color=colorlist[2], alpha=1)
ax[26].plot(np.arange(1, 11), q[:, 3], color=colorlist[3], alpha=1)
ax[26].plot([1, 10], [0.05, 0.05], ':', color='black')

ax[3].imshow(np.mean(pad_true, axis=0)[nt]/nt, origin='lower',  extent=[-10,
                                                                        10, 0, 1], aspect='auto', vmin=0, vmax=1, cmap=cmap, rasterized=True)
ax[9].imshow(np.mean(pad_false, axis=0)[nt]/(maxcomparisons-nt), origin='lower',  extent=[-10,
                                                                                          10, 0, 1], aspect='auto', vmin=0.00001, vmax=0.1, norm=LogNorm(), cmap=cmap, rasterized=True)
a = np.mean(pad_false, axis=0)[
    nt]/(np.mean(pad_false, axis=0)[nt]+np.mean(pad_true, axis=0)[nt])
ax[15].imshow(a, origin='lower',  extent=[-10, 10, 0, 1], aspect='auto',
              vmin=0.0001, vmax=1, norm=LogNorm(), cmap=cmap, rasterized=True)
q = fdr_per_quadrant(a)
bar = ax[21].bar(['Q1', 'Q2', 'Q3', 'Q4'], q, color='darkgray')
for i, b in enumerate(bar):
    b.set_color(colorlist[i])
ax[21].plot([-0.5, 3.5], [0.05, 0.05], ':', color='black')

q = []
for nti in range(1, 11):
    a = np.mean(pad_false, axis=0)[
        nti]/(np.mean(pad_false, axis=0)[nti]+np.mean(pad_true, axis=0)[nti])
    q.append(fdr_per_quadrant(a))
q = np.array(q)
ax[27].plot(np.arange(1, 11), q[:, 0], color=colorlist[0], alpha=1)
ax[27].plot(np.arange(1, 11), q[:, 1], color=colorlist[1], alpha=1)
ax[27].plot(np.arange(1, 11), q[:, 2], color=colorlist[2], alpha=1)
ax[27].plot(np.arange(1, 11), q[:, 3], color=colorlist[3], alpha=1)
ax[27].plot([1, 10], [0.05, 0.05], ':', color='black')

ax[4].imshow(np.mean(pas_true, axis=0)[nt]/nt, origin='lower',  extent=[-10,
                                                                        10, 0, 1], aspect='auto', vmin=0, vmax=1, cmap=cmap, rasterized=True)
ax[10].imshow(np.mean(pas_false, axis=0)[nt]/(maxcomparisons-nt), origin='lower',  extent=[-10,
                                                                                           10, 0, 1], aspect='auto', vmin=0.00001, vmax=0.1, norm=LogNorm(), cmap=cmap, rasterized=True)
a = np.mean(pas_false, axis=0)[
    nt]/(np.mean(pas_false, axis=0)[nt]+np.mean(pas_true, axis=0)[nt])
ax[16].imshow(a, origin='lower',  extent=[-10, 10, 0, 1], aspect='auto',
              vmin=0.0001, vmax=1, norm=LogNorm(), cmap=cmap, rasterized=True)
q = fdr_per_quadrant(a)
bar = ax[22].bar(['Q1', 'Q2', 'Q3', 'Q4'], q, color='darkgray')
for i, b in enumerate(bar):
    b.set_color(colorlist[i])
ax[22].plot([-0.5, 3.5], [0.05, 0.05], ':', color='black')

q = []
for nti in range(1, 11):
    a = np.mean(pas_false, axis=0)[
        nti]/(np.mean(pas_false, axis=0)[nti]+np.mean(pas_true, axis=0)[nti])
    q.append(fdr_per_quadrant(a))
q = np.array(q)
ax[28].plot(np.arange(1, 11), q[:, 0], color=colorlist[0], alpha=1)
ax[28].plot(np.arange(1, 11), q[:, 1], color=colorlist[1], alpha=1)
ax[28].plot(np.arange(1, 11), q[:, 2], color=colorlist[2], alpha=1)
ax[28].plot(np.arange(1, 11), q[:, 3], color=colorlist[3], alpha=1)
ax[28].plot([1, 10], [0.05, 0.05], ':', color='black')

ax[5].imshow(np.mean(pai_true, axis=0)[nt]/nt, origin='lower',  extent=[-10,
                                                                        10, 0, 1], aspect='auto', vmin=0, vmax=1, cmap=cmap, rasterized=True)

ax[11].imshow(np.mean(pai_false, axis=0)[nt]/(maxcomparisons-nt), origin='lower',  extent=[-10,
                                                                                           10, 0, 1], aspect='auto', vmin=0.00001, vmax=0.1, norm=LogNorm(), cmap=cmap, rasterized=True)
a = np.mean(pai_false, axis=0)[
    nt]/(np.mean(pai_false, axis=0)[nt]+np.mean(pai_true, axis=0)[nt])
ax[17].imshow(a, origin='lower',  extent=[-10, 10, 0, 1], aspect='auto',
              vmin=0.0001, vmax=1, norm=LogNorm(), cmap=cmap, rasterized=True)
q = fdr_per_quadrant(a)
bar = ax[23].bar(['Q1', 'Q2', 'Q3', 'Q4'], q, color='darkgray')
for i, b in enumerate(bar):
    b.set_color(colorlist[i])
ax[23].plot([-0.5, 3.5], [0.05, 0.05], ':', color='black')

q = []
for nti in range(1, 11):
    a = np.mean(pai_false, axis=0)[
        nti]/(np.mean(pai_false, axis=0)[nti]+np.mean(pai_true, axis=0)[nti])
    q.append(fdr_per_quadrant(a))
q = np.array(q)
ax[29].plot(np.arange(1, 11), q[:, 0], color=colorlist[0], alpha=1)
ax[29].plot(np.arange(1, 11), q[:, 1], color=colorlist[1], alpha=1)
ax[29].plot(np.arange(1, 11), q[:, 2], color=colorlist[2], alpha=1)
ax[29].plot(np.arange(1, 11), q[:, 3], color=colorlist[3], alpha=1)
ax[29].plot([1, 10], [0.05, 0.05], ':', color='black')


for i, a in enumerate(ax):
    a.set_ylim([0, 1])
    a.set_yticks([0, 0.25, 0.5, 0.75, 1])
    if i == 0:
        styler(a, aspectsquare=True,
               title=titles[i], ylabel='Covariance (truths)', xlabel=r'$\lambda$')
    elif i < 6:
        styler(a, aspectsquare=True, title=titles[i], xlabel=r'$\lambda$')
    elif i == 6 or i == 12:
        styler(a, aspectsquare=True,
               ylabel='Covariance (truths)', xlabel=r'$\lambda$')
    elif i < 18:
        styler(a, aspectsquare=True, xlabel=r'$\lambda$')
    elif i == 18:
        styler(a, aspectsquare=True, xticklabels=[
               'Q1', 'Q2', 'Q3', 'Q4'], ylabel='FDR')
    elif i > 18 and i < 24:
        styler(a, aspectsquare=True, xticklabels=['Q1', 'Q2', 'Q3', 'Q4'])
    elif i == 24:
        styler(a, aspectsquare=True, xlabel='True positives (%)',  ylabel='FDR')
    elif i >= 25:
        styler(a, aspectsquare=True, xlabel='True positives (%)')

plt.tight_layout()

fig.savefig(savedir + 'simulation2_results_tp_fp_fdr.svg')


# Plot a colorbar
from matplotlib.ticker import LogFormatterMathtext

fig, ax = plt.subplots(3, 1)
ax = ax.flatten()
pax = []
pax.append(ax[0].imshow(np.mean(uncorr_true, axis=0)[nt]/nt, origin='lower',
                        extent=[-10, 10, 0, 1], aspect='auto', vmin=0, vmax=1, cmap=cmap, rasterized=True))
pax.append(ax[1].imshow(np.mean(uncorr_false, axis=0)[nt]/(maxcomparisons-nt), origin='lower',
                        extent=[-10, 10, 0, 1], aspect='auto', vmin=0.00001, vmax=0.1, norm=LogNorm(), cmap=cmap, rasterized=True))
a = np.mean(uncorr_false, axis=0)[
    nt]/(np.mean(uncorr_false, axis=0)[nt]+np.mean(uncorr_true, axis=0)[nt])
pax.append(ax[2].imshow(a, origin='lower',  extent=[-10, 10, 0, 1], aspect='auto',
                        vmin=0.0001, vmax=1, norm=LogNorm(), cmap=cmap, rasterized=True))

for i, a in enumerate(ax):
    if i > 0:
        plt.colorbar(pax[i], format=LogFormatterMathtext())
    else:
        plt.colorbar(pax[i])

plt.tight_layout()
fig.savefig(savedir + 'simulation2_results_tp_fp_fdr_cbar.svg')
