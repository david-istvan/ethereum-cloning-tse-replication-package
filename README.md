# Replication package
### for the paper _Code Cloning in Smart Contracts on the Ethereum Platform: An Extended Replication Study_.

## About
This paper is an extended replication of the paper [_Code cloning in smart contracts: a case study on verified contracts from the Ethereum blockchain platform_](https://link.springer.com/article/10.1007/s10664-020-09852-5) by M. Kondo, G. Oliva, Z.M. Jiang, A. Hassan, and O. Mizuno. For the replication package of the original study, please, visit [https://github.com/SAILResearch/suppmaterial-18-masanari-smart_contract_cloning](https://github.com/SAILResearch/suppmaterial-18-masanari-smart_contract_cloning). For obtaining the corpus of 33,034 smart contracts, please, contact the authors of the original study.

## Contents

* `/01_data`
  * `/clonedata` – Results of the clone analysis by the [NiCad extension](https://github.com/eff-kay/nicad6) developed for this study.
    * `01_raw` – Raw results from the analysis.
    * `02_duplicates` – Duplicate data.
    * `03_openzeppelin.zip` – OpenZeppelin data. Requires unzipping into folder `03_openzeppelin`.
  * `/metadata` – Metadata about the authors, creation date and transactions of the contracts in the corpus.
  * `/prepared` - Prepared data for analysis. Contains potentially long-running scripts.
* `/02_analysis` - Analysis scripts.
* `/03_results` - Results.

## Reproduction

### Reproduction of the analyses

1. Clone this repository.
2. Install dependencies by running `pip install -r requirements.txt` in the root folder.
3. Extract `/01_data/clonedata/03_openzeppelin.zip` into folder `/01_data/clonedata/03_openzeppelin`, or run `python 01_unzip.py` in the `02_prepare` folder.
4. Run `python analysis.py` in the `/03_analysis` folder.
   1. Run `python analysis.py -o [observationId]` to run the analysis of a specific observation.
   2. Use the `-s` flag to stash the folder of the previous analyses.

### Reproduction of the prepared data (`/01_data/prepared`)

The prepared data is used in the analyses. The prepared data is included in this replication package in folder `/01_data/prepared`, but it can be reproduced from the raw data by following the steps below.

1. Run `python 03_mergeMetadata.py` in the `/02_prepare` folder.
2. Run `python 04_prepareAnalysisData.py` in the `/02_prepare` folder.
   1. Run `python 04_prepareAnalysisData.py -p [RQ or observation ID]` to prepare data for a specific RQ or observation.

Some preparation steps can take up to hours to complete. Please find the benchmarked execution times commented in the source code.