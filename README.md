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
├── run_visual.sh            # visualization shell script
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
nltk==3.7
gensim==4.2.0
scikit-learn==1.1.2
matplotlib==3.6.1
adjusttext==0.7.3
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

2.   Run **dataset/retrieve_dataset.py** by `sh dataset/retrieve_dataset.sh`. Modify **dataset/retrieve_dataset.sh** as needed.

     ```shell
     python dataset/retrieve_dataset.py \
         --data_dir=/path/to/tweet_dataset \
         --year=2019 \
         --months=1,2,3,4,5,6,7,8,9,10,11,12
     ```

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

1.   Run **run_word_freq.sh** to generate word frequency, tf-idf. Modify run_word_freq.sh as needed.

     ```shell
     python word_freq.py \
         --data_dir=/path/to/tweet_dataset \
         --year=2019,2020 \
         --month="10,11,12;1,2,3,4,5,6,7,8,9,10"
     ```

​		After running **run_word_freq.sh**, **vocab.csv** and **vocab_stat.csv** will be generated. vocab.csv contains the vocabulary. vocab.csv contains the vocabulary, word frequency and tf-idf information.

2.   Run **run_word2vec.sh** to generate word embedding. Modify run_word2vec.sh as needed.

     ```shell
     python word2vec.py \
         --data_dir=/path/to/tweet_dataset \
         --output_dir=/path/to/output_dir \
         --year=2019,2020 \
         --month="10,11,12;1,2,3,4,5,6,7,8,9,10" \
         --save_preprocessed_corpus
     ```

​		The model for each month will be saved to /output_dir/model. The embedding for each month will be saved to /output_dir/vector. Years, months, vocabulary and the merged embedding will be saved to word2vec_emb.plk, which will be used when plotting the distributional time series.

### Visualization

Run **run_visual.sh** to generate word embedding. Modify run_visual.sh as needed.

```shell
python visualization.py \
    --keyword_fpath=/path/to/keyword.txt \
    --model_path=/path/to/vector \
    --tf_fpath=/path/tp/vocab_stat.csv \
    --merged_emb_fpath=/path/to/word2vec_emb.plk \
    --output_dir=/path/to/plots \
    --year=2019,2020 \
    --month="10,12;2,4,6,8,10" \
    --ts_year=2019,2020 \
    --ts_month="10,11,12;1,2,3,4,5,6,7,8,9,10" \
    --plot_topn=5 \
    --vs_keyword="virtual" \
    --semantic_change_keyword="quarantine" \
    --tsne_keyword="panic,lonely,virtual,lockdown,quarantine"
```

| Argument                | Type     | Description                       |
| ----------------------- | -------- | --------------------------------- |
| vs_keyword              | str_list | Keywords for tf-idf vs w2v plot   |
| semantic_change_keyword | str_list | Keywords for semantic change plot |
| tsne_keyword            | str_list | Keywords for tsne plot            |

## Sample Output


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
