python train_ms2vec.py -s 3 -c "../../data/corpus/ares_dataset/extract_cluster_corpus/ares.cluster_word_with_dataset.txt" --ginitialize "random" --cinitialize "zeros"
python train_ms2vec.py -s 5 -c "../../data/corpus/ares_dataset/extract_cluster_corpus/ares.cluster_word_with_dataset.txt" --ginitialize "random" --cinitialize "zeros"
python train_ms2vec.py -s 3 -c "../../data/corpus/ares_dataset/extract_cluster_corpus/ares.cluster_lem_with_dataset.txt" --ginitialize "random" --cinitialize "zeros"
python train_ms2vec.py -s 5 -c "../../data/corpus/ares_dataset/extract_cluster_corpus/ares.cluster_lem_with_dataset.txt" --ginitialize "random" --cinitialize "zeros"
python train_ms2vec.py -s 3 -c "../../data/corpus/ares_dataset/extract_cluster_corpus/ares.cluster_word_with_dataset.txt" --ginitialize "random" --cinitialize "same"
python train_ms2vec.py -s 5 -c "../../data/corpus/ares_dataset/extract_cluster_corpus/ares.cluster_word_with_dataset.txt" --ginitialize "random" --cinitialize "same"
python train_ms2vec.py -s 3 -c "../../data/corpus/ares_dataset/extract_cluster_corpus/ares.cluster_lem_with_dataset.txt" --ginitialize "random" --cinitialize "same"
python train_ms2vec.py -s 5 -c "../../data/corpus/ares_dataset/extract_cluster_corpus/ares.cluster_lem_with_dataset.txt" --ginitialize "random" --cinitialize "same"