# python setup.py install を打ってから実行

# 時間関連
import time
import datetime
import pytz
import argparse

import sys
import logging

import os
import random
import numpy as np

# from gensim.models.word2vec import LineSentence # normal な gensim からもどちらからでも読み込める
# コーパスの順序等については，LineSentence の中身をいじる方が良いかもしれない
from ms2vec.ms2vec import LineSentence
from ms2vec.ms2vec import MSSG

import gensim

def print_log(text):
    global logging
    print(text)
    logging.info(text)

if __name__ == "__main__":    
    start_time = datetime.datetime.now(pytz.timezone('Asia/Tokyo')) 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sense_num", default = 3, type=int)
    parser.add_argument("-d", "--delimiter", default = "SENSE")
    parser.add_argument("-c", "--corpus_name",
                        default = "../../data/corpus/ares_dataset/extract_cluster_corpus/ares.cluster_word_with_dataset.txt")
    # random, zeros
    parser.add_argument("--ginitialize", default = "random")
    # same, zeros
    parser.add_argument("--cinitialize", default = "zeros")
    parser.add_argument("--pretrained_path", default = None)
    parser.add_argument("--seed", default = 0, type=int)
    args = parser.parse_args()

    # normal setting
    sense_num = args.sense_num
    if args.delimiter in {"POS", "SYNSET", "SENSE"}:
        delimiter = f"-{args.delimiter}-"
    else:
        delimiter = args.delimiter
    corpus_name = args.corpus_name
    
    global_initialize_method = args.ginitialize 
    cluster_initialize_method = args.cinitialize 
    pretrained_path = args.pretrained_path
    
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    
    min_count = 5
    vector_size = 300
    epochs = 5
    negative_num = 5
    window = 5
    skip_oov = False #True
    vector_size = 300
    max_sentence_length = 10**4

    np_value = -1
    min_sensecount = min_count
    clustering_method = "dynamic_kmeans"

    if "MSSG" not in os.listdir("../../data/models"):
        os.mkdir("../../data/models/MSSG")
        
    if pretrained_path is None:
        output_vectorname = f"../../data/models/MSSG/MSSG_sense_{sense_num}_ginitial_{global_initialize_method}_cinitial_{cluster_initialize_method}"
    else:
        output_vectorname = f"../../data/models/MSSG/MSSG_sense_{sense_num}_pretrained_{os.path.basename(pretrained_path)}"
        
    if "main_log" not in os.listdir():
        os.mkdir("main_log")
    main_logfile = f"main_log/main_log_{os.path.basename(output_vectorname)}.log"
    if "analysis_log" not in os.listdir():
        os.mkdir("analysis_log")
        
    analysis_logfile = f"analysis_log/analyze_topic_{os.path.basename(output_vectorname)}.log"
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
                        level=logging.INFO, filename=main_logfile)
    
    print_log(f"output_vectorname {output_vectorname}")
    print_log("running {}".format(" ".join(sys.argv)))
    print_log("start_time {}".format(start_time))  
    memo = "dot, select center in context's centers"
    print_log("Information {}".format(memo))     
    
    print_log("corpus {}".format(corpus_name))
    print_log("min_count {}".format(min_count))
    print_log("vector_size {}".format(vector_size))
    print_log("epochs {}".format(epochs))
    print_log("negative_num {}".format(negative_num))
    print_log("window {}".format(window))
    print_log("skip_oov {}".format(skip_oov))
    print_log("max_sentence_length {}".format(max_sentence_length))
    
    print_log("sense_num {}".format(sense_num))
    print_log("np_value {}".format(np_value))
    print_log("min_sense_count {}".format(min_sensecount))
    print_log("global_initialize_method {}".format(global_initialize_method))
    print_log("cluster_initialize_method {}".format(cluster_initialize_method))
    print_log("clustering method {}".format(clustering_method))
    print_log("analysis_logfile {}".format(analysis_logfile))

    corpus = LineSentence(corpus_name)    
    
    model = MSSG(corpus, sg=1, negative=negative_num, workers=4, epochs=epochs, window=window,
                 min_count=min_count, vector_size=vector_size, seed=seed, skip_oov=skip_oov,
                 sense_num=sense_num, delimiter=delimiter, min_sensecount=min_sensecount,
                 global_initialize=global_initialize_method, cluster_initialize=cluster_initialize_method, 
                 clustering_method=clustering_method, pretrained_path=pretrained_path,
                 np_value=np_value, sorted_vocab=0, analysis_logfile=analysis_logfile)
    
    end_time = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    print_log("end_time {}".format(end_time))  
    print_log("duration time {}".format(end_time-start_time))
                
    model_name = output_vectorname
    model.wv.save_word2vec_format(f"{model_name}.txt", binary=False)
    model.wv.save_sense2vec_format(f"{model_name}_sensevectors.txt", binary=False, min_cut_count = 0)
    model.wv.save_sense2vec_format(f"{model_name}_sensevectors_mincut_{min_count}.txt", binary=False, min_cut_count = min_count)
#     model.save(model_name)

    model = gensim.models.KeyedVectors.load_word2vec_format(f"{model_name}_sensevectors.txt", binary=False)
    for word in ["the", "mouse", "keyboard"]:
        for i in range(sense_num):
            wordname = word if i == 0 else word + delimiter + str(i)
            print_log(" ".join(["print nearest neighbors", word, wordname]))
            if wordname in set(model.index_to_key):
                print_log("\n".join([str(pair) for pair in model.most_similar(wordname)]))
