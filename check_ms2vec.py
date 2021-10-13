# python setup.py install を打ってから実行

# 時間関連
import time
import datetime
import pytz

import sys
import logging

import random
import numpy as np
seed = 0
random.seed(seed)
np.random.seed(seed)

# from gensim.models.word2vec import LineSentence # normal な gensim からもどちらからでも読み込める
# コーパスの順序等については，LineSentence の中身をいじる方が良いかもしれない
from ms2vec.ms2vec import LineSentence
from ms2vec.ms2vec import Word2Vec

def print_log(text):
    global logging
    print(text)
    logging.info(text)

if __name__ == "__main__":    
    start_time = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))  
    main_logfile = "main_log/main_log_{}.log".format(start_time)
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
                        level=logging.INFO, filename=main_logfile)
    print_log("running {}".format(" ".join(sys.argv)))
    print_log("start_time {}".format(start_time))  
    memo = "dot, select center in context's centers"
    print_log("Information {}".format(memo))     
    
    # normal setting
    corpus_name = "text8"
    min_count = 10
    vector_size = 300
    epochs = 2
    negative_num = 5
    window = 5
    skip_oov=True
    vector_size = 300
    max_sentence_length = 100
    output_vectorname = "models/ms2vec_{}".format(start_time)
    
    
    print_log("corpus {}".format(corpus_name))
    print_log("min_count {}".format(min_count))
    print_log("vector_size {}".format(vector_size))
    print_log("epochs {}".format(epochs))
    print_log("negative_num {}".format(negative_num))
    print_log("window {}".format(window))
    print_log("skip_oov {}".format(skip_oov))
    print_log("max_sentence_length {}".format(max_sentence_length))
    print_log("output_vectorname {}".format(output_vectorname))
    
    # specific setting
    sense_num = 3
    delimiter = "--"
    np_value = -1
    min_sensecount = 100
    global_initialize_method = "random" # random, zeros
    cluster_initialize_method = "zeros" # same, zeros
    clustering_method = "dynamic_kmeans"
    analysis_logfile = "analysis_log/analyze_topic_{}.log".format(start_time)
    
    print_log("sense_num {}".format(sense_num))
    print_log("np_value {}".format(np_value))
    print_log("min_sense_count {}".format(min_sensecount))
    print_log("global_initialize_method {}".format(global_initialize_method))
    print_log("cluster_initialize_method {}".format(cluster_initialize_method))
    print_log("clustering method {}".format(clustering_method))
    print_log("analysis_logfile {}".format(analysis_logfile))
             
    corpus = LineSentence(corpus_name, max_sentence_length = max_sentence_length)    
    
    model = Word2Vec(corpus, sg=1, negative=negative_num, workers=4, epochs=epochs, window=window,
                     min_count=min_count, vector_size=vector_size, seed=seed, skip_oov=skip_oov,
                     sense_num=sense_num, delimiter=delimiter, min_sensecount=min_sensecount,
                     global_initialize=global_initialize_method, cluster_initialize=cluster_initialize_method, 
                     clustering_method=clustering_method,
                     np_value=np_value, sorted_vocab=0, analysis_logfile=analysis_logfile)
    
    end_time = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    print_log("end_time {}".format(end_time))  
    print_log("duration time {}".format(end_time-start_time))

    for word in ["the", "mouse", "keyboard"]:
        for i in range(sense_num):
            wordname = word if i == 0 else word + delimiter + str(i)
            print_log(" ".join(["print nearest neighbors", word, wordname]))
            if wordname in set(model.wv.index_to_key):
                print_log("\n".join([str(pair) for pair in model.wv.most_similar(wordname)]))
                
    model_name = output_vectorname
    model.wv.save_word2vec_format(model_name+".txt", binary=False)
    model.wv.save_sense2vec_format(model_name+"_sensevectors.txt", binary=False, min_cut_count = 0)
    model.wv.save_sense2vec_format(model_name+"_sensevectors_mincut{}.txt".format(min_count), binary=False, min_cut_count = min_count)
    model.save(model_name)
