# Replication package
### for the paper _Code Cloning in Smart Contracts: A Replication Study_.

### Contents

* `/01_corpus` - Corpus of verified smart contracts used in this study.
* `/02_authordata` - Author information of the contracts of the corpus.
* `/03_clones` - Results of the clone analysis by the [NiCad extension](https://github.com/eff-kay/nicad6) developed for this study.
* `/04_staged_data` - Prepared and staged data for analysis.
* `/05_analysis` - Analysis of data.

### Reproduction of analysis results

1. Clone this repository.
2. Run the analysis script.

### Reproducing the pre-staged data

1. Clone this repository.
2. Run the `/04_staged_data/01_mergeFullData.py` script.
3. Run the `/04_staged_data/02_prepareAnalysisData.py` script.
   1. List the observation IDs in `observationsToPrepareDataFor` to prepare the data for. Use `observationsToPrepareDataFor = allObservations` to generate data for all observations. (Might run for hours.)

### Reproducing the clone analysis

Refer to the [NiCad extension](https://github.com/eff-kay/nicad6) developed for this study.
