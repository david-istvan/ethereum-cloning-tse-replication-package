# Replication package
### for the paper _Code Cloning in Smart Contracts: A Replication Study_.

### Contents

* `/01_corpus` - Corpus of verified smart contracts used in this study.
* `/02_authordata` - Author information of the contracts of the corpus.
* `/03_clones` - Results of the clone analysis by the [NiCad extension](https://github.com/eff-kay/nicad6) developed for this study.
* `/04_staged_data` - Prepared and staged data for analysis.
* `/05_analysis` - Analysis of data.
* `/06_results` - Results.

### Reproduction of analysis results

1. Clone this repository.
2. Run `python analysis.py` the `/05_analysis` folder.

### Reproducing the pre-staged data

1. Clone this repository.
2. Run `python 01_mergeFullData.py` in the `/04_staged_data` folder.
3. Run `python 02_prepareAnalysisData.py` in the `/04_staged_data` folder.
   1. List the observation IDs in `observationsToPrepareDataFor` to prepare the data for. Use `observationsToPrepareDataFor = allObservations` to generate data for all observations. (Might run for hours.)

### Reproducing the clone analysis

Refer to the [NiCad extension](https://github.com/eff-kay/nicad6) developed for this study.
