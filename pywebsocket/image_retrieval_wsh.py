# Copyright 2011, Google Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#     * Neither the name of Google Inc. nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


_GOODBYE_MESSAGE = u'Goodbye'

# from gensim import models
import numpy as np
import operator
import base64


# Set configuration
class Config():
    def __init__(self):
        self.text_model_type = "glove"
        self.text_model_name = "glove_model_InstaCities1M.txt"
        self.img_embeddings_name = "triplet_softNegativeBatch_m50_notNormalize_frozen_glove_tfidf_SM_iter_260000"
        self.num_topics = 400
        self.num_results = 14

# Loads Word2Vec or GloVe model
def load_text_model(cfg):
    if cfg.text_model_type == "word2vec":
        print("Need to import gensim!")
        # text_model = models.Word2Vec.load("../data/models/word2vec/" + cfg.text_model_name)
    elif cfg.text_model_type == "glove":
        text_model = {}
        for line in open("../data/models/glove/" + cfg.text_model_name,'r'):
            d = line.split(',')
            values = np.zeros(cfg.num_topics)
            for t in range(0, cfg.num_topics):
                values[t] = d[t + 1]
            text_model[d[0]] = values
    else:
        print("text_model_type " + str(cfg.text_model_type) + " not recognized")
        return 0
    return text_model

# Loads precomputed image embeddings
def load_img_embeddings(cfg):
    database = {}
    file = open("../data/img_embeddings/" + cfg.img_embeddings_name + ".txt", "r")
    for line in file:
        d = line.split(",")
        regression_values = np.zeros(cfg.num_topics)
        for t in range(0,cfg.num_topics):
            regression_values[t] = d[t + 1]
        database[d[0]] = regression_values / sum(regression_values)
    return database

# Infers word embedding
def get_word_embedding(word, model, cfg):
    if cfg.text_model_type == "word2vec":
        embedding = model[word]
    elif cfg.text_model_type == "glove":
        embedding = model[word]
    else:
        print("text_model_type " + str(cfg.text_model_type) + " not recognized")
        return 0
    embedding = embedding - min(embedding)
    embedding = embedding / sum(embedding)
    return embedding

# Gets word or image embedding
def get_embedding(query, text_model, database, cfg):
    if "/" not in query:
        query_embedding = get_word_embedding(query, text_model, cfg)
    else:
        query_embedding = database[query]
    return query_embedding

print("Initializing")
cfg = Config()
print(" --> Loading text model")
text_model = load_text_model(cfg)
print(" --> Loading image embeddings")
database = load_img_embeddings(cfg)
print("Ready!")


def web_socket_do_extra_handshake(request):
    # This example handler accepts any request. See origin_check_wsh.py for how
    # to reject access from untrusted scripts based on origin value.

    pass  # Always accept.


def web_socket_transfer_data(request):
    while True:
        full_query = request.ws_stream.receive_message()
        if full_query is None:
            return
        if isinstance(full_query, unicode):
            print("Message received: " + full_query)
            # Got a query, return image ids
            if ';query' in full_query:
                full_query = full_query.replace(' ','').replace(';query','')
                print(" --> Query: " + str(full_query))
                print(" --> Computing query embeddings")
                if "+" not in full_query and "-" not in full_query:
                    query_embedding = get_embedding(full_query, text_model, database, cfg)

                elif "+" in full_query:
                    queries = full_query.split("+")
                    query_embedding = np.zeros(cfg.num_topics)
                    for q in queries:
                        query_embedding+=get_embedding(q, text_model, database, cfg)
                    query_embedding /= len(queries)
                elif "-" in full_query:
                    queries = full_query.split("-")
                    query_embedding = get_embedding(queries[0], text_model, database, cfg)
                    queries.pop(0)
                    for q in queries:
                        query_embedding-=get_embedding(q, text_model, database, cfg)

                if "-" in full_query and "+" in full_query:
                    print("Warning: Cannot handle multiple different operators")

                print(" --> Searching")
                distances = {}
                for id in database:
                    distances[id] = np.dot(database[id], query_embedding)
                distances = sorted(distances.items(), key=operator.itemgetter(1), reverse=True)
                filenames = ""
                for idx, id in enumerate(distances):
                    if idx == 0: filenames += "data/retrieval_img/" + cfg.text_model_type + '/' + id[0] + ".jpg"
                    else: filenames += ";data/retrieval_img/" + cfg.text_model_type + '/' + id[0] + ".jpg"
                    if idx == cfg.num_results - 1: break
                print("Done!")

                request.ws_stream.send_message(filenames, binary=False)

            # Got an img id, return img content
            else:
                with open('/home/raulgomez/datasets/MMSemanticRetrievalDemo/' + full_query, "rb") as imageFile:
                    img_base64 = base64.b64encode(imageFile.read())
                # print(" --> Image read.")
                request.ws_stream.send_message(img_base64, binary=False)

            if full_query == _GOODBYE_MESSAGE:
                return
        else:
            return

