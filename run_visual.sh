python visualization.py \
    --keyword_fpath=/path/to/keyword.txt \
    --model_path=/path/to/vector \
    --tf_fpath=/path/to/vocab_stat.csv \
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
