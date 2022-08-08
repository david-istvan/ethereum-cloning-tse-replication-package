# Replication package

### for the paper _Code Cloning in Smart Contracts on the Ethereum Platform: An Extended Replication Study_.

## About

This paper is an extended replication of the paper [_Code cloning in smart contracts: a case study on verified contracts from the Ethereum blockchain platform_](https://link.springer.com/article/10.1007/s10664-020-09852-5) by M. Kondo, G. Oliva, Z.M. Jiang, A. Hassan, and O. Mizuno. For the replication package of the original study, please, visit [https://github.com/SAILResearch/suppmaterial-18-masanari-smart_contract_cloning](https://github.com/SAILResearch/suppmaterial-18-masanari-smart_contract_cloning). To obtain the corpus of 33,034 smart contracts, please, contact the authors of the original study.

## Contents

- `/01_data`
  - `/clonedata` – Results of the clone analysis by the [NiCad extension](https://github.com/eff-kay/nicad6) developed for this study.
    - `/raw` – Raw results from the analysis.
    - `/duplicates` – Cleaned data.
    - `openzeppelin.zip` – OpenZeppelin data. Requires unzipping into folder `openzeppelin`.
  - `/metadata` – Metadata about the authors, creation date and transactions of the contracts in the corpus.
  - `/prepared` - Prepared pickle files for data analysis.
- `/02_prepare` - Scripts for preparing the data in `/01_data/prepared`. Contains potentially long-running scripts. In such cases, the approximate execution times are reported in the source files.
- `/03_analysis` - Analysis scripts for the automated analysis of data.
- `/04_results` - Results of the analyses, including charts and numeric results. Some of these results are discussed in the paper in great detail. Every analysis result corresponds to a particular observation in the paper, clearly identified in the name of the generated observation file.

## Reproduction

The following describes four reproduction scenarios. Any of the scenarios can be executed independently from the others.
* [Reproduction of the analyses](#reproduction-of-the-analyses): reproduces the analysis results in `/04_results`, including charts and numeric results. The scripts use the prepared data contained in the `/01_data/prepared` folder.
* [Reproduction of the prepared data](#reproduction-of-the-prepared-data-01_dataprepared): reproduces the prepared data in `/01_data/prepared` by (i) merging author, transaction and file length metadata into the clone data; and (ii), pre-processing data for analysis and persisting the pre-processed data into pickle files. Some of the pre-processing steps are potentially time-consuming. In such cases, the approximate execution times are reported in the source file.
* [Reproduction of the cleaned data](#reproduction-of-the-cleaned-data-01_dataclonedataduplicates): reproduces the cleaned data in `/01_data/clonedata/duplicates` from the raw data in `/01_data/clonedata/raw` by bringing the contents of the `.xml` files into a consolidated form.
* [Reproduction of the raw data](#reproducing-the-clone-analysis-01_dataclonedataraw): reproduces the raw data `/01_data/clonedata/raw` by running the [NiCad extension](https://github.com/eff-kay/nicad6) developed for this study.
 
**NOTE:** The following steps have been tested with `python>=3.7 && python<3.10`.

### Reproduction of the analyses

Follow the steps below to reproduce the analysis results in `/04_results`, including charts and numeric results. The scripts use the prepared data contained in the /01_data/prepared folder.

1. Clone this repository.
2. Install dependencies by running `pip install -r requirements.txt` in the root folder.
3. Extract `/01_data/clonedata/openzeppelin.zip` into folder `/01_data/clonedata/openzeppelin`, or run `python 01_unzip.py` in the `02_prepare` folder.
4. Run `python analysis.py` in the `/03_analysis` folder.
   1. Run `python analysis.py -o [observationId]` to run the analysis of a specific observation.
   2. Use the `-s` flag to stash the folder of the previous analyses.

### Reproduction of the prepared data (`/01_data/prepared`)

Follow the steps below to reproduce the prepared data in `/01_data/prepared` by (i) merging author, transaction and file length metadata into the clone data; and (ii), pre-processing data for analysis and persisting the pre-processed data into pickle files. Some of the pre-processing steps are potentially time-consuming. In such cases, the approximate execution times are reported in the source file.

1. Run `python 03_mergeMetadata.py` in the `/02_prepare` folder.
2. Run `python 04_prepareAnalysisData.py` in the `/02_prepare` folder.
   1. Run `python 04_prepareAnalysisData.py -p [RQ or observation ID]` to prepare data for a specific RQ or observation.

Some preparation steps can take up to hours to complete. Please find the benchmarked execution times commented in the source code.

### Reproduction of the cleaned data (`/01_data/clonedata/duplicates`)

Follow the steps below to reproduce the cleaned data in `/01_data/clonedata/duplicates` from the raw data in `/01_data/clonedata/raw` by bringing the contents of the `.xml` files into a consolidated form.

The cleaned data is used in the data preparation scripts. The cleaned data is included in this replication package in folder `/01_data/clonedata/duplicates`, but it can be reproduced from the raw data by following the steps below.

1. Run `python 02_cleanup.py` in the `/02_prepare` folder.

### Reproducing the clone analysis (`/01_data/clonedata/raw`)

Follow the steps below to reproduce the raw clone data in `/01_data/clonedata/raw` by running the [NiCad extension](https://github.com/eff-kay/nicad6) developed for this study.

To obtain the corpus of 33,034 smart contracts, please, contact the authors of the [original study](https://github.com/SAILResearch/suppmaterial-18-masanari-smart_contract_cloning).

A Docker image is maintained on [Docker Hub](https://hub.docker.com/repository/docker/faizank/nicad6) and can be obtained by running: `docker pull faizank/nicad6:TSE`.

The following process assumes [docker](https://docs.docker.com/get-started/) is installed and working correctly, and the image is pulled. You can verify that image by issuing `docker images` from the terminal and see that there is an image named `faizank/nicad6` available in the list.

**NOTE:** The following steps have been tested with `docker_engine==20.10.17(build==100c701)`

1. Create a new folder `/systems/source-code` and move the corpus to this folder.
2. Create a new folder `/output` to store the result of clone analysis.
3. Execute the analysis by issuing the following command: `docker run --platform linux/x86_64 -v output:/nicad6/01_data -v systems:/nicad6/systems faizank/nicad6`. This will generate the output artefacts inside the `output` folder.
4. Move the contents of the `/output` folder to `/01_data` and use the python scripts discussed above for the rest of the replication.

Should you prefer to build the image from scratch, please, refer to the repository of the [NiCad extension](https://github.com/eff-kay/nicad6) developed for this study.

### Further experimentation with the tool

To experiment with the tool, issue `docker run --platform linux/x86_64 -v output:/nicad6/01_data -v systems:/nicad6/systems -it faizank/nicad6 bash`.
