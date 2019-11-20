from multipy.fwer import bonferroni
from multipy.fdr import lsu
import numpy as np
from scipy.stats import pearsonr
import sys

def spent_alpha(alpha=0.05, alpha_history=None):
    if alpha_history is None:
        return alpha
    return alpha - np.sum(alpha_history)

def invest_alpha(p, invest_rate=0.5, alpha=0.05, wealth=0.05):
    if not isinstance(wealth, list):
        wealth = [wealth]
    alpha_j = (wealth[-1] * invest_rate)
    p = p < alpha_j
    if p == True:
        # Check this should not be alpha_j.
        wealth.append(wealth[-1] + alpha)
    else:
        wealth.append(wealth[-1] - (alpha_j)/(1-alpha_j))

    return p, wealth

def gen_data_sim(npts, maxcomparisons, numbertrue=5, truecorr=0.5, likelihood_of_truth=0):
    """
    This function generates the data for both simulation 1 and simulation 2. 
    """
    # likelihood_of_truth if 0 truths are equally likely, if 1 truths are twice as likely to be drawn (100% more likely)
    # if lot is negative, then the reverse occurs (-1 entails falsehoods are twice as likely
    truecomp = []
    cov = np.zeros([maxcomparisons+1, maxcomparisons+1])
    cov[:numbertrue+1, :numbertrue+1] = truecorr
    cov[np.arange(numbertrue+1), np.arange(numbertrue+1)] = 1
    mu = np.zeros(maxcomparisons+1)
    draw_data = np.random.multivariate_normal(
        mu, cov, (npts)).astype(np.double)
    y = draw_data[:, 0]
    x = draw_data[:, 1:]

    if numbertrue > 0:
        if likelihood_of_truth < 0:
            c_prob_norm = np.array([1, -likelihood_of_truth+1])
        else:
            c_prob_norm = np.array([likelihood_of_truth+1, 1])

        c_prob_norm = c_prob_norm/np.sum(c_prob_norm)
        c_prob = np.zeros(maxcomparisons)
        c_prob[:numbertrue] = c_prob_norm[0]/numbertrue
        c_prob[numbertrue:] = c_prob_norm[1]/(maxcomparisons-numbertrue)
        c = np.random.choice(maxcomparisons, maxcomparisons,
                             p=c_prob, replace=False)
        truecomp = np.vstack([np.where(c == n)[0]
                              for n in np.arange(numbertrue)])
        x = x[:, c]
    else:
        truecomp = []

    return x, y, np.array(truecomp)


#PARAMETERS THAT MUST BE SET
# Where to save figures
figpath = '/home/william/work/datasetdecay/'  
# Where to save the data
datadir = '/scratch/users/wiltho/data/datasetdecay/sequentialcomparisons/'
# The simulation run number. 
# Code was run on slurm so this was specfieid externally. This can be made into a for loop instead. 
# r also sets the numpy seed. 
r = int(sys.argv[1])

## Simulation parameters
maxcomparisons = 100  # Test from 1-maxcomparisons
alpha_alternatives = [0.05, 0.01, 0.001]  # Test for different alpha thresholds
npts = 100  # Number of data point in collrection
probrange = np.arange(-10, 10) # In the paper, this is the \lambda 
trrange = np.arange(0, 1, 0.025) # The true covariance of vairables. 
trueposrange = np.arange(0, 11) # Number of true positives (truepos = 0 is simulation 1, 1-10 is simulation 2)

# Preassign different output values. 
# pai = alpha investing 
# pas = alpha spending 
# pad = alpha det 
# bon = bonferroni 
# fdr = fdr
# uncorr = uncorrcted 
# null = SIM1 (i.e no true positives)
# true = SIM2 performance on "true" variables (i.e. underlying correlation)
# false = SIM2 performance on "false" variables (i.e. no underlying correlation)
pai_true = np.zeros([11, len(trrange), len(probrange)])
pai_false = np.zeros([11, len(trrange), len(probrange)])
pas_true = np.zeros([11, len(trrange), len(probrange)])
pas_false = np.zeros([11, len(trrange), len(probrange)])
pad_true = np.zeros([11, len(trrange), len(probrange)])
pad_false = np.zeros([11, len(trrange), len(probrange)])
bon_true = np.zeros([11, len(trrange), len(probrange)])
bon_false = np.zeros([11, len(trrange), len(probrange)])
fdr_true = np.zeros([11, len(trrange), len(probrange)])
fdr_false = np.zeros([11, len(trrange), len(probrange)])
uncorr_true = np.zeros([11, len(trrange), len(probrange)])
uncorr_false = np.zeros([11, len(trrange), len(probrange)])

# Set random seed
np.random.seed(r)

# Run a for loop over (1): number of truths (nt), (2) lambda (prob); 3 the true covariance value (tr).  
for nti, nt in enumerate(trueposrange):
    print(nt)
    for pi, prob in enumerate(probrange):
        print(prob)
        for tri, tr in enumerate(trrange):
            # Generate the data for this parameter configuration. 
            x, y, true_trials = gen_data_sim(
                npts, maxcomparisons, numbertrue=nt, truecorr=tr, likelihood_of_truth=prob)
            # Calculate pairwise correlations between the all the comparisons 
            ry = np.zeros([maxcomparisons])
            py = np.zeros([maxcomparisons])
            for n in range(maxcomparisons):
                ry[n], py[n] = pearsonr(y, x[:, n])
            # pSimultaneous bonferroni: 
            p_bon = bonferroni(py)
            # Simultaneous 
            p_fdr = lsu(py)
            # uncorrected
            p_uncorr = py < 0.05
            # alpha debt
            p_pad = np.array([p < (0.05/(i+1)) for i, p in enumerate(py)])
            # alpha spending 
            p_alphaspend = np.zeros([maxcomparisons])
            for n in range(maxcomparisons):
                if n == 0:
                    alpha_history = None
                alpha = spent_alpha(alpha_history=alpha_history)*0.5
                p_alphaspend[n] = py[n] < alpha
                if n == 0:
                    alpha_history = [0.025]
                else:
                    alpha_history.append(alpha)
            # alpha investing
            p_alphainvest = np.zeros([maxcomparisons])
            wealth_fun = np.zeros([maxcomparisons])
            for n in range(maxcomparisons):
                if n == 0:
                    wealth = 0.05
                p, wealth = invest_alpha(py[n], wealth=wealth)

                p_alphainvest[n] = p
            wealth_fun[:] = wealth[:-1]
            # Add data to output variables 
            false_trials = np.arange(maxcomparisons)
            if len(true_trials) > 0:
                false_trials = np.delete(false_trials, true_trials)
            if nt != 0:
                bon_true[nti, tri, pi] = p_bon[true_trials].sum()
                fdr_true[nti, tri, pi] = p_fdr[true_trials].sum()
                uncorr_true[nti, tri, pi] = p_uncorr[true_trials].sum()
                pad_true[nti, tri, pi] = p_pad[true_trials].sum()
            elif nt == 0 and pi == 0 and tri == 0:
                bon_null = p_bon
                fdr_null = p_fdr
                uncorr_null = p_uncorr
                pad_null = p_pad
            bon_false[nti, tri, pi] = p_bon[false_trials].sum()
            fdr_false[nti, tri, pi] = p_fdr[false_trials].sum()
            uncorr_false[nti, tri, pi] = p_uncorr[false_trials].sum()
            pad_false[nti, tri, pi] = p_pad[false_trials].sum()
            if nt != 0:
                pai_true[nti, tri, pi] = p_alphainvest[true_trials].sum()
                pas_true[nti, tri, pi] = p_alphaspend[true_trials].sum()
            elif nt == 0 and pi == 0 and tri == 0:
                pai_null = p_alphainvest
                pas_null = p_alphaspend

            pai_false[nti, tri, pi] = p_alphainvest[false_trials].sum()
            pas_false[nti, tri, pi] = p_alphaspend[false_trials].sum()

# Save sim1 data.
np.save(datadir + 'uncorr_null_' + str(r), uncorr_null)
np.save(datadir + 'bon_null_' + str(r), bon_null)
np.save(datadir + 'fdr_null_' + str(r), fdr_null)
np.save(datadir + 'pad_null_' + str(r), pad_null)
np.save(datadir + 'pas_null_' + str(r), pas_null)
np.save(datadir + 'pai_null_' + str(r), pai_null)
np.save(datadir + 'uncorr_true_' + str(r), uncorr_true)
np.save(datadir + 'uncorr_false_' + str(r), uncorr_false)
np.save(datadir + 'bon_true_' + str(r), bon_true)
np.save(datadir + 'bon_false_' + str(r), bon_false)
np.save(datadir + 'fdr_true_' + str(r), fdr_true)
np.save(datadir + 'fdr_false_' + str(r), fdr_false)
np.save(datadir + 'pad_true_' + str(r), pad_true)
np.save(datadir + 'pad_false_' + str(r), pad_false)
np.save(datadir + 'pas_true_' + str(r), pas_true)
np.save(datadir + 'pas_false_' + str(r), pas_false)
np.save(datadir + 'pai_true_' + str(r), pai_true)
np.save(datadir + 'pai_false_' + str(r), pai_false)
