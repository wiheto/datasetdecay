#%% 
# import data 

import pandas as pd
import scipy.stats as stats
import numpy as np
import statsmodels.api as sm
from multipy.fwer import bonferroni
from multipy.fdr import lsu
import matplotlib.pyplot as plt
from plotje import styler as make_pretty_ax
np.random.seed(2019)
#%% 
# specify paths and load data
# Path to the data
figdir = './'
datadir = './'

#%% 
df = pd.read_csv(datadir + '/hcp_unrestricted_data.csv')
# hcp_variablesofinterest contains 188 variables that were selected as possible variables that could have been an analysis, 6 of these did not have any data associated with them, so it gets wittled down to 182. 
voi = pd.read_csv(datadir + '/hcp_variablesofinterest.csv',sep='\t')

fs = voi['fs'].dropna()
behav = voi['behav'].dropna()

# fs contains all the freesurfer output (thinckness, area, curv). Only select thickness for this analysis
fs_thick = []
for _, x in fs.iteritems():
    if x.endswith('_Thck'): 
        fs_thick.append(x)
fs_thick = pd.Series(fs_thick)

#%%
# Standardize values. 
behav_z = pd.DataFrame()
for i, n in behav.iteritems(): 
    behav_z[n] = (df[n] - df[n].mean()) / df[n].std()

fs_z = pd.DataFrame()
for i, n in fs_thick.iteritems(): 
    fs_z[n] = (df[n] - df[n].mean()) / df[n].std()

fs_z = fs_z.dropna(how='all')



#%% 
# Run comparisions 
# Preallovcate
r = np.zeros([len(fs_thick), len(behav)])
p = np.zeros([len(fs_thick), len(behav)])
# Loop through each behaviour and fit OLS model of all thickness estimates to it.
p = []
preg = []
blist = []
for j, b in enumerate(behav_z.iteritems()):
        behavtmp = b[1].dropna()
        xtmp = fs_z.loc[list(set(fs_z.index).intersection(behavtmp.index))]
        behavtmp = behavtmp.loc[list(set(fs_z.index).intersection(behavtmp.index))]

        if len(xtmp) > 0:
                reg = sm.OLS(behavtmp, xtmp)
                regfit = reg.fit()
                p.append(regfit.pvalues.values)
                blist.append(b[0])

df = pd.DataFrame(data={'Behaviour/personality measure': blist})
df.to_csv(datadir + '/TableS1.csv')
p = np.array(p).transpose() 

## 
# Run comparisions 
# Preallocate
r = np.zeros([len(fs_thick), len(behav)])
p1 = np.zeros([len(fs_thick), len(behav)])
p2 = np.zeros([len(fs_thick), len(behav)])
# Loop through each behaviour and fit OLS model of all thickness estimates to it.


#%%  


# the alpha_spend and alpha invest functions were written independently of get_number_findings function and were never combined. This could have been done better.  

# Note, the 
def alpha_spend(alpha=0.05, alpha_history=None):
    if alpha_history is None: 
        return alpha
    return alpha - np.sum(alpha_history) 
    

def invest_alpha_with_fdr(p, invest_rate=0.5, alpha=0.05, wealth=0.05):
    if not isinstance(wealth, list): 
        wealth = [wealth]
    alpha_j = (wealth[-1] * invest_rate)        
    p = lsu(p, q=alpha_j)
    # If there is any significant simultneous finding, increase alpha. If not, decrease. 
    if sum(p) > 0: 
        # Check this should not be alpha_j.
        wealth.append(wealth[-1] + alpha)
    else: 
        wealth.append(wealth[-1] - (alpha_j)/(1-alpha_j))

    return p, wealth 


def get_number_findings(p, method, threshold=0.05, output='findings', permutations=100):
    """
    This function allows for alpha debt, and simultaneous corrections. It can output the number of studies, findings and (for simultaneous corrections) which studies were significant

    Parameters
    -----------
    p : array 
        p-values (dependent, independent variables)
    method : str
        either fdr, fwe or debt
    threshold : alpha threshold
    output : str
        findings (number of significant findings), studies (number of DVs with at least one significant finding) where (which dings were significant), or debt (all results for alpha debt)
    permutations: int
        number of permutations (alpha debt only)
    
    Output: 
    -------
    If output == 'findings': findings_sequentially_uncorrected, findings_allsimultaneously_corrected
    If output == 'studies': studies_sequentially_uncorrected, studies_allsimultaneously_corrected    
    If output == 'debt': findings_debt_randomodr, findings_debt_informedodr, studies_debt_randomodr, studies_debt_informedodr
    If output == 'where': which "allsimultaneously" findings were significant (output = [DV,IV])    
    """
    if method == 'fdr':
        accumuluate_fdr = 0 
        pub_studies_acum = 0
        for j in range(p.shape[1]):
            accum = np.sum(lsu(p[:,j], q=threshold)) 
            accumuluate_fdr += accum 
            if accum>0:
                pub_studies_acum += 1
            if j > 0: 
                psim = np.random.beta(1,10,j*p.shape[0])
                afdr_tmp = lsu(np.hstack([psim, p[:, j]]), q=threshold)[-p.shape[0]:]
            else: 
                afdr_tmp = p[:, j] < threshold
            accum = np.sum(afdr_tmp) 
        all_fdr_ps = lsu(p.flatten(), q=threshold).reshape(p.shape)
        pub_studies_all = np.sum(np.sum(all_fdr_ps,axis=0)>0)
        all_fdr = sum(lsu(p.flatten(), q=threshold))
        if output == 'findings':
                return accumuluate_fdr, all_fdr, 
        if output == 'studies': 
                return pub_studies_acum, pub_studies_all
        if output == 'where': 
            return np.where(all_fdr_ps==True)

    elif method == 'fwe' or method == 'debt':
        accumuluate_bonferroni = 0 
        pub_studies_acum = 0
        for j in range(p.shape[1]):
            accum = np.sum(bonferroni(p[:,j], alpha=threshold)) 
            accumuluate_bonferroni += accum 
            if accum>0:
                pub_studies_acum += 1
        all_bonferroni_flat = bonferroni(p.flatten(), alpha=threshold)
        all_bonferroni = np.sum(all_bonferroni_flat)

        all_bonferroni_rs = all_bonferroni_flat.reshape(p.shape)
        pub_studies_all = np.sum(np.sum(all_bonferroni_rs,axis=0)>0)


    if method == 'debt':
        informed_studies = np.where(all_bonferroni_rs==True)[1]
        accumuluate_correct_debt = np.zeros(permutations)
        accumuluate_correct_debt_informend = np.zeros(permutations)
        pub_studies_acum = 0
        pub_studies_debt = np.zeros(permutations)
        pub_studies_debt_informed = np.zeros(permutations)
        for c in range(permutations):
            pshuffle = p.copy()
            odr = np.random.permutation(pshuffle.shape[1])
            pshuffle = pshuffle[:, odr] 
            for j in range(p.shape[1]):
                corraccum = np.sum(lsu(pshuffle[:,j].flatten(), q=threshold/(j+1))) 
                accumuluate_correct_debt[c] += corraccum 
                if corraccum > 0:
                    pub_studies_debt[c] += 1
            pshuffle = p.copy()
            # Lambda = 0 (50/50 an informed comparison is drawn, at each draw)
            cs = np.zeros(182) + (2/4)/180
            cs[informed_studies] = (2/4)/2
            odr = np.random.choice(182, 182, replace=False, p=cs)
            pshuffle = pshuffle[:, odr] 
            for j in range(p.shape[1]):
                corraccum = np.sum(lsu(pshuffle[:,j].flatten(), q=threshold/(j+1))) 
                accumuluate_correct_debt_informend[c] += corraccum 
                if corraccum > 0:
                    pub_studies_debt_informed[c] += 1


    if output == 'findings':
            return accumuluate_bonferroni, all_bonferroni
    if output == 'studies': 
            return pub_studies_acum, pub_studies_all
    if output == 'debt': 
            return accumuluate_correct_debt, accumuluate_correct_debt_informend, pub_studies_debt, pub_studies_debt_informed
    if output == 'where': 
        return np.where(all_bonferroni_rs==True)

#%%
# Call get_number_findings

fdr_uncorrected, fdr_corrected = get_number_findings(p, 'fdr', 0.05)
fwe_uncorrected, fwe_corrected = get_number_findings(p, 'fwe', 0.05)

studies_fdr_uncorrected, studies_fdr_corrected = get_number_findings(p, 'fdr', 0.05, 'studies')
studies_fwe_uncorrected, studies_fwe_corrected = get_number_findings(p, 'fwe', 0.05, 'studies')

findings_pad_fdr, findings_pad_fdr_informed, studies_pad_fdr, studies_pad_fdr_informed = get_number_findings(p, 'debt', 0.05, 'debt')

which_fdr_studies = get_number_findings(p, 'fdr', 0.05, 'where')
which_fwe_studies = get_number_findings(p, 'fwe', 0.05, 'where')

print(which_fdr_studies)
print(which_fwe_studies)

informed_studies = which_fdr_studies[1]

#%%

# These functions do the same as get_number_findings, but with alpha spending and alpha investing which were not integrated with that funciton 
# Spending, random order
permutations = 100
pas_fdr = np.zeros([68,182,permutations])
for n in range(permutations):
        pshuffle = p.copy()
        odr = np.random.permutation(pshuffle.shape[1])
        pshuffle = pshuffle[:, odr] 
        alpha_history = None
        for j in range(pshuffle.shape[1]):        
                alpha = alpha_spend(alpha_history=alpha_history)*0.5
                pas_fdr[:,odr[j],n] = pshuffle[:, j] < lsu(pshuffle[:, j], q=alpha) 
                if alpha_history is None: 
                        alpha_history = [0.025]
                else: 
                        alpha_history.append(alpha)     

# Spending, informed order. 
permutations = 100
pas_fdr_informed = np.zeros([68,182,permutations])
for n in range(permutations):
        pshuffle = p.copy()
        c = np.zeros(182) + (2/4)/180
        # odds of the two significant comparisions are 
        c[[39,40]] = (2/4)/2
        odr = np.random.choice(182, 182, replace=False, p=c)
        pshuffle = pshuffle[:, odr] 
        alpha_history = None
        for j in range(pshuffle.shape[1]):        
                alpha = alpha_spend(alpha_history=alpha_history)*0.5
                pas_fdr_informed[:,odr[j],n] = pshuffle[:, j] < lsu(pshuffle[:, j], q=alpha) 
                if alpha_history is None: 
                        alpha_history = [0.025]
                else: 
                        alpha_history.append(alpha)     

# Investing, random order
permutations = 100
pai_fdr = np.zeros([68, 182, permutations])
for n in range(permutations):
        pshuffle = p.copy()
        odr = np.random.permutation(pshuffle.shape[1])
        pshuffle = pshuffle[:, odr] 
        wealth = 0.05
        for j in range(pshuffle.shape[1]):
                pai_fdr[:,odr[j],n], wealth = invest_alpha_with_fdr(pshuffle[:,j], wealth=wealth)

# Investing, informed order
pai_fdr_informed = np.zeros([68, 182, permutations])
for n in range(permutations):
        pshuffle = p.copy()
        c = np.zeros(182) + (2/4)/180
        c[informed_studies] = (2/4)/2
        odr = np.random.choice(182, 182, replace=False, p=c)
        pshuffle = pshuffle[:, odr] 
        wealth = 0.05
        for j in range(pshuffle.shape[1]):
                pai_fdr_informed[:,odr[j],n], wealth = invest_alpha_with_fdr(pshuffle[:,j], wealth=wealth)


#%% 
# Get the number of "findings" for each method

findings_pas_fdr = pas_fdr.mean(axis=-1).sum()
findings_pas_fdr_informed = pas_fdr_informed.mean(axis=-1).sum()
findings_pai_fdr = pai_fdr.mean(axis=-1).sum()
findings_pai_fdr_informed = pai_fdr_informed.mean(axis=-1).sum()

findings_pas_fdr_min = pas_fdr.sum(axis=0).sum(axis=0).min()
findings_pas_fdr_max = pas_fdr.sum(axis=0).sum(axis=0).max()
findings_pas_fdr_informed_min = pas_fdr_informed.sum(axis=0).sum(axis=0).min()
findings_pas_fdr_informed_max = pas_fdr_informed.sum(axis=0).sum(axis=0).max()
findings_pai_fdr_min = pai_fdr.sum(axis=0).sum(axis=0).min()
findings_pai_fdr_max = pai_fdr.sum(axis=0).sum(axis=0).max()
findings_pai_fdr_informed_min = pai_fdr_informed.sum(axis=0).sum(axis=0).min()
findings_pai_fdr_informed_max = pai_fdr_informed.sum(axis=0).sum(axis=0).max()

findings_pas_fdr_std = pas_fdr.sum(axis=0).sum(axis=0).std()
findings_pas_fdr_informed_std = pas_fdr_informed.sum(axis=0).sum(axis=0).std()
findings_pai_fdr_std = pai_fdr.sum(axis=0).sum(axis=0).std()
findings_pai_fdr_informed_std = pai_fdr_informed.sum(axis=0).sum(axis=0).std()

#%%
# Get the number of studies 

studies_pas_fdr = np.mean(np.sum(pas_fdr>0,axis=0)>0,axis=-1).sum()
studies_pas_fdr_informed = np.mean(np.sum(pas_fdr_informed>0,axis=0)>0,axis=-1).sum()
studies_pai_fdr = np.mean(np.sum(pai_fdr>0,axis=0)>0,axis=-1).sum()
studies_pai_fdr_informed = np.mean(np.sum(pai_fdr_informed>0,axis=0)>0,axis=-1).sum()

studies_pas_fdr_min = (pas_fdr.sum(axis=0)>0).sum(axis=0).min()
studies_pas_fdr_max = (pas_fdr.sum(axis=0)>0).sum(axis=0).max()
studies_pas_fdr_informed_min = (pas_fdr_informed.sum(axis=0)>0).sum(axis=0).min()
studies_pas_fdr_informed_max = (pas_fdr_informed.sum(axis=0)>0).sum(axis=0).max()
studies_pai_fdr_min = (pai_fdr.sum(axis=0)>0).sum(axis=0).min()
studies_pai_fdr_max = (pai_fdr.sum(axis=0)>0).sum(axis=0).max()
studies_pai_fdr_informed_min = (pai_fdr_informed.sum(axis=0)>0).sum(axis=0).min()
studies_pai_fdr_informed_max = (pai_fdr_informed.sum(axis=0)>0).sum(axis=0).max()

studies_pas_fdr_std = (pas_fdr.sum(axis=0)>0).sum(axis=0).std()
studies_pas_fdr_informed_std = (pas_fdr_informed.sum(axis=0)>0).sum(axis=0).std()
studies_pai_fdr_std = (pai_fdr.sum(axis=0)>0).sum(axis=0).std()
studies_pai_fdr_informed_std = (pai_fdr_informed.sum(axis=0)>0).sum(axis=0).std()



#%% 
# For sanity, make sure that the fdr and fwe are finding the same significant results (or FWE is a superset of FDR)
which_fdr_studies = get_number_findings(p, 'fdr', 0.05, 'where')
which_fwe_studies = get_number_findings(p, 'fwe', 0.05, 'where')

# All the values in this list:
print(which_fdr_studies)
# Should be found in this list: 
print(which_fwe_studies)


#%% 
# Plot results

labels = ['Bonferroni (all)', 'FDR (all)', '-', 'Bonferroni (within study)', 'FDR (within study)',  '--', r'$\alpha$-debt', r'$\alpha$-spending', r'$\alpha$-investing', '---', r'$\alpha$-debt (informed)', r'$\alpha$-spending (informed)', r'$\alpha$-investing (informed)',]
findings = [fwe_corrected, fdr_corrected, 0, fwe_uncorrected, fdr_uncorrected, 0, findings_pad_fdr.mean(),  findings_pas_fdr, findings_pai_fdr, 0, findings_pad_fdr_informed.mean(),  findings_pas_fdr_informed, findings_pai_fdr_informed,]
studies = [studies_fwe_corrected, studies_fdr_corrected, 0, studies_fwe_uncorrected, studies_fdr_uncorrected, 0, studies_pad_fdr.mean(),  studies_pas_fdr, studies_pai_fdr, 0, studies_pad_fdr_informed.mean(),  studies_pas_fdr_informed, studies_pai_fdr_informed.mean(),]
#studies = [studies_fdr_uncorrected, studies_fdr_corrected, 0, studies_fwe_uncorrected, studies_fwe_corrected, studies_fwe_accumulate.mean(), studies_fwe_accumulate.min(), studies_fwe_accumulate.max()]

labelscut = [r'$\alpha$-debt', r'$\alpha$-spending', r'$\alpha$-investing', r'$\alpha$-debt (informed)', r'$\alpha$-spending (informed)', r'$\alpha$-investing (informed)',]
maxfindings = np.array([findings_pad_fdr.max(),  findings_pas_fdr_max, findings_pai_fdr_max, findings_pad_fdr_informed.max(),  findings_pas_fdr_informed_max, findings_pai_fdr_informed_max,])
minfindings = np.array([findings_pad_fdr.min(),  findings_pas_fdr_min, findings_pai_fdr_min, findings_pad_fdr_informed.min(),  findings_pas_fdr_informed_min, findings_pai_fdr_informed_min])
findingscut = np.array([findings_pad_fdr.mean(),  findings_pas_fdr, findings_pai_fdr, findings_pad_fdr_informed.mean(),  findings_pas_fdr_informed, findings_pai_fdr_informed])
stdfindings = np.array([findings_pad_fdr.std(),  findings_pas_fdr_std, findings_pai_fdr_std, findings_pad_fdr_informed.std(),  findings_pas_fdr_informed_std, findings_pai_fdr_informed_std])
maxstudies = np.array([studies_pad_fdr.max(),  studies_pas_fdr_max, studies_pai_fdr_max, studies_pad_fdr_informed.max(),  studies_pas_fdr_informed_max, studies_pai_fdr_informed_max,])
minstudies = np.array([studies_pad_fdr.min(),  studies_pas_fdr_min, studies_pai_fdr_min, studies_pad_fdr_informed.min(),  studies_pas_fdr_informed_min, studies_pai_fdr_informed_min,])
studiescut = np.array([studies_pad_fdr.mean(),  studies_pas_fdr, studies_pai_fdr, studies_pad_fdr_informed.mean(),  studies_pas_fdr_informed, studies_pai_fdr_informed,])
stdstudies = np.array([studies_pad_fdr.std(),  studies_pas_fdr_std, studies_pai_fdr_std, studies_pad_fdr_informed.std(),  studies_pas_fdr_informed_std, studies_pai_fdr_informed_std])


fig, ax = plt.subplots(1, 2, sharex=True, figsize=(10,4), sharey=True)
ax = ax.flatten()
xlabs = ['# Significant tests', '# Studies (with at least one finding)']

ax[0].barh(labels, findings, color='darkgray',zorder=5)
ax[0].scatter(maxfindings, labelscut, color='black', s=5, marker='x',zorder=10)
ax[0].scatter(minfindings, labelscut, color='black', s=5, marker='x',zorder=10)
ax[0].errorbar(findingscut, labelscut, xerr=stdfindings, fmt='x', color='black',zorder=20)


ax[1].barh(labels, studies, color='darkgray')
ax[1].scatter(maxstudies, labelscut, color='black', s=5, marker='x',zorder=10)
ax[1].scatter(minstudies, labelscut, color='black', s=5, marker='x',zorder=10)
ax[1].errorbar(studiescut, labelscut, xerr=stdstudies, fmt='x', color='black',zorder=20)

ax[0].plot([2,2],[-0.5,12.5],linestyle='--',zorder=30)
ax[1].plot([2,2],[-0.5,12.5],linestyle='--',zorder=30)

ax[0].set_xlim([0,45])
ax[0].set_xticks(np.arange(0,41,10))
for i, a in enumerate(ax): 
        make_pretty_ax(a, yticklabels=labels, xlabel=xlabs[i], rotatexticks=False)


plt.tight_layout()
plt.gca().invert_yaxis()
fig.savefig(figdir + '/empiricial_example.svg')
fig.savefig(figdir + '/empiricial_example.png')
