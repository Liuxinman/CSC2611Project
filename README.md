# Measuring and Visualizing Short-Term Change in Word Representation and Usage Due to Coronavirus Pandemic in Twitter

This is a course project for CSC2611 (Computational Models of Semantic Change Fall 2022). 

This work will investigate the short-term word meaning shifts over time using a corpus of Twitter data
during the COVID-19 pandemic. Specifically, the short-term changes in word contextualized representation
(representation shift) will be measured and visualized in contrast with such changes in word frequency (concept drift).

## Directory Information

```txt
.
├── README.md                # Documentation
├── dataset 
│   ├── merge_dataset.py     # Merge daily data into monthly data
│   ├── retrieve_dataset.py  # Retrieve tweets data from Twitter
│   └── retrieve_dataset.sh  # Shell script for retrieving tweets data from Twitter
├── preprocess.py            # Tweet proprocessor 
├── run_word2vec.sh          # Word2vec training shell script
├── run_word_freq.sh         # Word freq and tf-idf training shell script
├── utils.py                 # Utility functions
├── visualization.py         # Visualization
├── word2vec.py              # Generate distributional time series
└── word_freq.py             # Generate word frequency and tf-idf time series
```

## Getting Started

### Environment

```shell
python==3.9.13
pandas==1.5.0
numpy==1.23.3
nltk==3.7
gensim==4.2.0
scikit-learn==1.1.2
```

### Dataset

1.   To use Twitter full-archive API to access data from Twitter, please first apply Academic Research access level. Then, modify  `.env`  file to enter your private keys and tokens.

     ```shell
     API_KEY=...
     API_KEY_SECRET=...
     ACCESS_TOKEN=...
     ACCESS_TOKEN_SECRET=...
     BEARER_TOKEN=...
     ```

2.   Run **dataset/retrieve_dataset.py** by `sh dataset/retrieve_dataset.sh`. 

     ```
     python dataset/retrieve_dataset.py \
         --data_dir=/path/to/tweet_dataset \
         --year=2019 \
         --months=1,2,3,4,5,6,7,8,9,10,11,12
     ```

     Modify **dataset/retrieve_dataset.sh** as needed.

     | Argument | Type     | Description                            |
     | -------- | -------- | -------------------------------------- |
     | data_dir | str      | data directory path to save tweet data |
     | year     | int      | year of data to retrieve               |
     | months   | int_list | months in the year of data to retrieve |

     The generated data directory looks as follows:
     ```txt
     .
     ├── 2019
         ├── 10
         │   ├── 1.csv
         │   ├── 2.csv
         │   ├── ...
         │   ├── 30.csv
         │   └── 31.csv
         ├── 11
         │   ├── 1.csv
         │   ├── 2.csv
         │   ├── ...
         │   ├── 29.csv
         │   └── 30.csv
         ├── 12
             ├── 1.csv
             ├── 2.csv
             ├── ...
             ├── 30.csv
             └──  31.csv
     ```

3.   Run **dataset/merge_dataset.py** to merge daily data into monthly data. Modify `dataset_dir` in **dataset/merge_dataset.py** as needed.

     ```txt
     .
     ├── 2019_merged
         ├── 10_merged.csv
         ├── 11_merged.csv
         └── 12_merged.csv
     ```

### Time Series



### Visualization



## Reference

The idea of this project is draw from the following papers.

```latex
@article{russian, 
    title={Measuring, Predicting and Visualizing Short-Term Change in Word Representation and Usage in VKontakte Social Network}, 
    volume={11}, 
    url={https://ojs.aaai.org/index.php/ICWSM/article/view/14938},
    number={1}, 
    journal={Proceedings of the International AAAI Conference on Web and Social Media}, 
    author={Stewart, Ian and Arendt, Dustin and Bell, Eric and Volkova, Svitlana}, 
    year={2017}, 
    month={May}, 
    pages={672-675} 
}
```

```latex
@inproceedings{del2019short-term,
	Author = {Del Tredici, Marco and Fern\'andez, Raquel and Boleda, Gemma},
	Booktitle = {Proceedings of NAACL-HLT 2019 (Annual Conference of the North American Chapter of the Association for Computational Linguistics)},
	Title = {{Short-Term Meaning Shift: A Distributional Exploration}},
	Year = {2019}
  }
```
