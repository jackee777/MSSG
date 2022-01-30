import gensim

ginitial = "random"
cinitial = "zeros" # same
min_count = 5
embedding_name = f"../../data/models/MSSG/MSSG_ginitial_{ginitial}_cinitial_{cinitial}_sensevectors_mincut{min_count}.txt"
model = gensim.models.KeyedVectors.load_word2vec_format(embedding_name, binary=False)

sense_num = 3
delimiter = "-SENSE-"

for word in ["the", "mouse", "keyboard"]:
    for i in range(sense_num):
        wordname = word if i == 0 else word + delimiter + str(i)
        print(" ".join(["print nearest neighbors", word, wordname]))
        if wordname in set(model.index_to_key):
            print("\n".join([str(pair) for pair in model.most_similar(wordname)]))


