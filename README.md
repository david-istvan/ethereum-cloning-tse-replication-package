# Replication package
### for the paper _Code Cloning in Smart Contracts on the Ethereum Platform: An Extended Replication Study_.

## About
This paper is an extended replication of the paper [_Code cloning in smart contracts: a case study on verified contracts from the Ethereum blockchain platform_](https://link.springer.com/article/10.1007/s10664-020-09852-5) by M. Kondo, G. Oliva, Z.M. Jiang, A. Hassan, and O. Mizuno. For the replication package of the original study, please visit [https://github.com/SAILResearch/suppmaterial-18-masanari-smart_contract_cloning](https://github.com/SAILResearch/suppmaterial-18-masanari-smart_contract_cloning).

## Contents

* `/01_corpus` - Corpus of verified smart contracts used in this study. Identical to the corpus of the original study, available from its [replication package](https://github.com/SAILResearch/suppmaterial-18-masanari-smart_contract_cloning).
* `/02_metadata` - Metadata about the authors, creation date and transactions of the contracts in the corpus.
* `/03_clones` - Results of the clone analysis by the [NiCad extension](https://github.com/eff-kay/nicad6) developed for this study.
* `/04_staged_data` - Prepared and staged data for analysis.
* `/05_analysis` - Automated analysis scripts.
* `/06_results` - Results.

## Reproduction

### Reproduction of the analyses

1. Clone this repository.
2. Download the data file from [TODO].
3. Install dependencies by running `pip install -r requirements.txt` in the root folder.
4. Run `python analysis.py` in the `/05_analysis` folder.
   1. Run `python analysis.py -o [observationId]` to run the analysis of a specific observation.
   2. Use the `-s` flag to stash the folder of the previous analyses.

### Reproduction of the pre-staged data (`/04_staged_data`)

The pre-staged data is used in the analyses. The pre-staged data is included in the data package. Follow the steps below to reproduce the pre-staged data from the cleaned raw clone data.

1. Clone this repository.
2. Download the data file from [TODO].
3. Install dependencies by running `pip install -r requirements.txt` in the root folder.
4. Run `python 01_mergeMetadata.py` in the `/04_staged_data` folder.
5. Run `python 02_prepareAnalysisData.py` in the `/04_staged_data` folder.
   1. List the observation IDs in `observationsToPrepareDataFor` to prepare the data for. Use `observationsToPrepareDataFor = allObservations` to generate data for all observations.
   2. `prepareObservation3()` (producing `data_observation3.p`) is an especially time-consuming script and might run for up to 2.5 hours.
   
### Reproduction of the cleaned data from the raw data (`/03_clones`)

The cleaned data is used in the pre-processing scripts generating the pre-staged data. The cleaned data is included in the data package. Follow the steps below to reproduce the cleaned data from the raw clone data.

1. Clone this repository.
2. Download the data file from [TODO].
3. Install dependencies by running `pip install -r requirements.txt` in the root folder.
4. Run `python 00_cleanup.py` in the `/04_staged_data` folder.
5. Follow the reproduction steps of the pre-staged data.

### Reproducing the clone analysis (`/03_clones`)

Refer to the repository of the [NiCad extension](https://github.com/eff-kay/nicad6) developed for this study.
