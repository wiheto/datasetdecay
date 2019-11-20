## DATA REQUIRED

1. Create an account/log in at: [the human connectome](http://db.humanconnectome.org)
2. Once logged in, download the 1200 unrestricted behavioural data file. [Quicklink](https://db.humanconnectome.org/REST/search/dict/Subject%20Information/results?format=csv&removeDelimitersFromFieldValues=true&restricted=0&project=HCP_1200)
3. The downloaded file will get its own name based on the access date and username. Rename it to "hcp_unrestricted_data.csv" and place it in this directory. 

## SOFTWARE REQUIREMENTS

Python 3.5+, numpy, scipy, pandas, matplotlib, multipy, statsmodels, (For the figure style you also need plotje).

## CODE EXECUTION ORDER

### Simulations

1. Change the necessary path variables in run_simulations.py (line 64 and 66).
2. Run by `python run_simulations.py <iteration>`. Where iteration number is an integer. This will then run one simulation iteration and save the output in the `datadir` directory defined on line 66. This was run 1000 times using a slurm script with the iterations number between 0-999.
3. Change the necessary path variables in `plot_simulations.py` (line 8 and 10).
4. If you ran a different number of simulation iterations, then this has to be amended in the loading of the data. 
4. Run `python plot_simuluations.py` and the figures will be saved in the `savedir` directory on line 10.

_Output_ 

1. `run_simulations` will save numpy files in `datadir`. 
2. `plot_simuluations` will save 3 figures. 

### Empirical data

1. `figdir` and `datadir` need to be amended in line 16 and 17
2. Run by `python empirical_example.py`

_Output_

1. A figure in `figdir`
2. A csv file containing the 182 variables called TableS1.csv in `datadir`.

### Other files

hcp_variablesofinterest.csv is a file containing two columns (`fs` and `behav`). Each entry references in a column in hcp_unrestricted_data.csv.
`fs` are the 68 freesurfer cortical thickness measures used. `behav` are 188 psychological and behavoiural measures that, from their description, sounded like varraible someone might consider interesting to cororelate with brain anatomy (prioritizing any age adjusted values when they were present). 6 of these variables did resulted in 0 columns being selected in empirical_example.py and were dropped, leaving 182 variables. 