# Run the parameter optimization process
(parameter-optimization)=

Parameter optimization is the process by which the nonlinear `subplex` optimization algorithm [(see Rowan, 1990)](https://dl.acm.org/doi/10.5555/100816) is used to set the parameters of the onset detection algorithms. The objective function was to maximize the mean [F-score](wiki:F-score) obtained between the algorithm results and a ground truth set of reference annotations (contained in `.\references\manual_annotations`), for a given percentage (10%) of the dataset. The optimization process sets different parameters for detecting onsets foreach instrument (piano, bass, drums), and for detecting beats in the mixed audio signal.

```{note} Creating annotation files
:class: dropdown
The reference annotations were created with the `Sonic Visualiser` [(Cannam et al., 2010)](https://www.sonicvisualiser.org/doc/reference/4.0.1/en/) software by the lead author and two undergraduate research assistants, and all were checked by the lead author. If you're interested in creating your own annotations to use as part of this process, see {ref}`Create new manual annotations <create-annotations>`.
```

```{warning}
We only support the use of the optimized parameter set we created ourselves, (i.e., the `.csv` and `.json` files contained in `.\references\parameter_optimisation` by deafult). If you choose to re-run the parameter optimization yourself or use your own ground truth annotations, we cannot guarentee that you'll be able to reproduce our results.
```

## Setting up

First, ensure that you've followed the installation instructions in {ref}`Building the database <build-database>` up to the {ref}`Onset detection <build-database-detect>` section. 

## Running the parameter optimization

Inside the virtual environment for the project you created when building the database, run the following command:

```
python src\detect\optimize_detection_parameters.py
>>> optimizing onset detection for piano, bass, drums ...
>>> optimising parameters across 30 track/instrument combinations ...
>>> ... instrument piano, iteration 0/?, mean F: 0.6402, stdev F: 0.0977, 30 tracks (0 loaded from cache),
```

This command will start the program off on optimizing the detection parameters for each instrument in turn, before optimizing the beat tracking algorithm. You'll notice that the mean and standard deviation F-score obtained for each iteration of the algorithm is printed directly to the console.

By default, the program will optimize detection for the `.\references\corpus_chronology.xlsx` corpus file. To change this, you can pass in the `-corpus_fname` flag to direct to another file in this folder, like:

```
python src\detect\optimize_detection_parameters.py -corpus_fname corpus_bill_evans
>>> optimizing onset detection for piano, bass, drums ...
```

```{tip}
On any given iteration step, if the program detects that a particular combination of parameters have already been tried for an instrument, it will skip this iteration and load the corresponding F-scores directly from disk (contained in `.\references\parameter_optimisation\{corpus_fname}\*.csv` files). To suppress this functionality, delete or rename the `.csv` files inside `.\references\parameter_optimisation\{corpus_fname}`.
```

:::{dropdown} Troubleshooting
Expect the optimization process to take a great deal of time to complete. You can expect each iteration step for an instrument (piano, bass, drums) to take ~90-120 seconds, depending on the combination of parameters being tested. For the beat detection on the audio mixture, an iteration step may take upwards of 20 minutes, due to the greater computational demand of this algorithm.

To end the optimization process early (not recommended), you can pass in the `-maxtime` flag followed by a number of seconds (`int`), or the `-maxeval` flag followed by a number of iterations (`int`), after which the algorithm will be made to converge.

To disable optimization for either onset or beat detection, pass in `-optimize_stems False` or `-optimize_mix False`, respectively. 

Finally, the program will (by default) make use of every available CPU core for optimization. To control this behaviour, pass in the `-n_jobs` flag, followed by the number of cores to use (`int`).
:::

## Check the results

Once the optimization process has completed, you can find the converged parameters inside `.\references\parameter_optimisation\{corpus_fname}\converged_parameters.json`. These will be then be used by the relevant algorithms when running `.\src\detect\detect_onsets.py`. 

:::{dropdown} Inspect iteration results

You can also inspect the results at every iteration of the optimization process by checking the individual `.csv` files inside `.\references\parameter_optimisation\{corpus_fname}`. Each row of these files corresponds to the results obtained from a single track. You can load these `.csv` files into your Python session with Pandas, e.g.:

```
import pandas as pd
from src import utils

res = pd.read_csv(rf'{utils.get_project_root()}\references\parameter_optimisation\{corpus_fname}\onset_detect_piano.csv')
```

To see the average F-score for each iteration step (i.e., the objective function the algorithm was attempting to maximize):

```
grp = res.groupby('iterations').mean(numeric_only=True)
grp['f_score']
```

To see the final set of optimized parameters (i.e., the iteration step with the highest average F-score):

```
max_idx = grp['f_score'].idxmax()
grp.iloc[max_idx]
```

In most cases, this will be the final iteration.
:::