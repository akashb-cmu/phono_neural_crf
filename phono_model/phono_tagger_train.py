#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
# Must set theano flags before theano is imported
os.environ["THEANO_FLAGS"] = "optimizer=None,mode=FAST_RUN,floatX=float64,exception_verbosity=high"
import numpy as np
import optparse
import itertools
from collections import OrderedDict
from utils import create_input, create_phono_model_input
import loader

from utils import models_path, eval_script, eval_temp, evaluate_phono_model
from loader import tag_mapping
from loader import update_tag_scheme
from phono_model import Model
import my_utils
import cPickle


# Read parameters from command line
optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="",
    help="Train set location"
)
optparser.add_option(
    "-d", "--dev", default="",
    help="Dev set location"
)
optparser.add_option(
    "-t", "--test", default="",
    help="Test set location"
)
optparser.add_option(
    "-s", "--tag_scheme", default="iobes",
    help="Tagging scheme (IOB or IOBES)"
)
optparser.add_option(
    "-l", "--lower", default="0",
    type='int', help="Whether the input word input word embeddings are for lower cased words. This "
                     "does NOT affect character embedding sequences."
)
optparser.add_option(
    "-z", "--zeros", default="0",
    type='int', help="Whether digits in words in the input word embeddings AND phono char embeddings have "
                     "been replaced with 0s. THIS ALSO AFFECTS CHARACTER EMBEDDING SEQUENCES."
)
optparser.add_option(
    "-w", "--word_dim", default="100",
    type='int', help="Token embedding dimension after projection from input word embedding size."
)
optparser.add_option(
    "-W", "--word_lstm_dim", default="100",
    type='int', help="Token LSTM hidden layer size"
)
optparser.add_option(
    "-B", "--word_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for words"
)
optparser.add_option(
    "-a", "--use_caps", default="1",
    type='int', help="Whether to use capitalization (0 to disable)"
)
optparser.add_option(
    "--use_cats", default="1",
    type='int', help="Whether to use character category features in the "
)
optparser.add_option(
    "-f", "--crf", default="1",
    type='int', help="Use CRF (0 to disable)"
)
optparser.add_option(
    "-D", "--dropout", default="0.5",
    type='float', help="Droupout on the input (0 = no dropout)"
)
optparser.add_option(
    "-L", "--lr_method", default="sgd-lr_.005",
    help="Learning method (SGD, Adadelta, Adam..)"
)
optparser.add_option(
    "-r", "--reload", default="0",
    type='int', help="Reload the last saved model"
)
optparser.add_option(
    "-N", "--model_name", default="garbage_model",
    type='str', help="Custom model name"
)

optparser.add_option(
    "--train_transfer_model", default=0,
    type='int', help="Whether to train a transfer model (ortho feats discarded)"
)
# May have to change the usage of this option when using Yulia's method

optparser.add_option(
    "--use_type_sparse_feats", default=0,
    type='int', help="Whether to use sparse features appended to inputs of the word LSTM",
)

optparser.add_option(
    "--use_token_sparse_feats", default=0,
    type='int', help="Whether to use sparse features appended to outputs of the word LSTM",
)

optparser.add_option(
    "--use_convolution", default=0,
    type='int', help="Whether to use convolution over char lstm output",
)

optparser.add_option(
    "--ortho_char_dim", default=25,
    type='int', help="Dimension of projected orthographic character vectors",
)

optparser.add_option(
     "--ortho_char_lstm_dim", default="25",
    type='int', help="Orthographic Char LSTM hidden layer size"
)

optparser.add_option(
    "--phono_char_dim", default="15",
    type='int', help="Phonological char embedding dimension"
)
optparser.add_option(
    "--phono_char_lstm_dim", default="15",
    type='int', help="Phonological Char LSTM hidden layer size"
)

optparser.add_option(
    "-b", "--char_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for chars"
)
optparser.add_option(
    "--type_sparse_feats_proj_dim", default="-1",
    type='int', help="Dimension to project type level sparse features to"
)
# TO DO :
# DEFAULT SHOULD BE SOME FACTOR OF THE ORIGINAL SPARSE FEATS PROJ DIM
optparser.add_option(
    "--token_sparse_feats_proj_dim", default="-1",
    type='int', help="Dimension to project token level sparse features to"
)

optparser.add_option(
    "--use_ortho_attention", default=0,
    type='int', help="Whether to use attention over orthographic characters in input to CRF",
)

optparser.add_option(
    "--use_phono_attention", default=0,
    type='int', help="Whether to use attention over phonological characters in input to CRF",
)

optparser.add_option(
    "--src_lang_cca_vec_location", default=None,
    type='str', help="Location of the source language word vectors. This option must be provided.",
)
# TO DO : Add the target language vectors location when implementing the transfer scenario
optparser.add_option(
    "--src_lang_epi_vec_files_path_prefix", default=None,
    type='str', help="Source language epitran vectors location. Not providing this will necessitate"
                     "extraction of epitran vectors which is time-consuming.",
)

optparser.add_option(
    "--src_lang", default="turkish",
    type='str', help="Source language",
)

optparser.add_option(
    "--eval_script", default=eval_script,
    type='str', help="Source language",
)

optparser.add_option(
    "--n_epochs", default=100,
    type='int', help="Number of epochs",
)

opts1 = optparser.parse_args()
opts = optparser.parse_args()[0]

# Parse parameters
parameters = OrderedDict()
n_epochs = opts.n_epochs


parameters['tag_scheme'] = opts.tag_scheme
parameters['lower'] = opts.lower == 1
parameters['zeros'] = opts.zeros == 1
parameters['char_bidirect'] = opts.char_bidirect == 1
parameters['word_dim'] = opts.word_dim
parameters['word_lstm_dim'] = opts.word_lstm_dim
parameters['word_bidirect'] = opts.word_bidirect == 1
parameters['use_caps'] = opts.use_caps == 1
parameters['use_cats'] = opts.use_cats == 1
parameters['crf'] = opts.crf == 1
parameters['dropout'] = opts.dropout
parameters['lr_method'] = opts.lr_method
parameters['model_name'] = opts.model_name
parameters["train_transfer_model"] = opts.train_transfer_model == 1
parameters["use_type_sparse_feats"] = opts.use_type_sparse_feats == 1
parameters["use_token_sparse_feats"] = opts.use_token_sparse_feats == 1
parameters["use_convolution"] = opts.use_convolution == 1
parameters["ortho_char_dim"] = opts.ortho_char_dim
parameters["ortho_char_lstm_dim"] = opts.ortho_char_lstm_dim
parameters["phono_char_dim"] = opts.phono_char_dim
parameters["phono_char_lstm_dim"] = opts.phono_char_lstm_dim
parameters["type_sparse_feats_proj_dim"] = opts.type_sparse_feats_proj_dim
parameters["token_sparse_feats_proj_dim"] = opts.token_sparse_feats_proj_dim
parameters["use_ortho_attention"] = opts.use_ortho_attention == 1
parameters["use_phono_attention"] = opts.use_phono_attention == 1
parameters["src_lang_cca_vec_location"] = opts.src_lang_cca_vec_location
parameters["src_lang_epi_vec_files_path_prefix"] = opts.src_lang_epi_vec_files_path_prefix
parameters["src_lang"] = opts.src_lang
parameters["src_lang_epi_vec_files_path_prefix"] = opts.src_lang_epi_vec_files_path_prefix
eval_script = opts.eval_script

src_epi = my_utils.lang_to_epi_dict[parameters["src_lang"]]

# Check parameters validity
assert parameters["use_type_sparse_feats"] is False, "Type sparse feats not implemented " \
                                                     "yet"
assert parameters["use_token_sparse_feats"] is False, "Token sparse feats not implemented " \
                                                     "yet"
assert parameters["train_transfer_model"] is False, "Transfer case not implemented yet"

assert parameters["src_lang_cca_vec_location"] is not None \
       and os.path.isfile(parameters["src_lang_cca_vec_location"]), \
    "src_lang_cca_vec_location not provided or oath is invalid"
assert os.path.isfile(opts.train)
assert os.path.isfile(opts.dev)
assert os.path.isfile(opts.test)
assert parameters['ortho_char_dim'] > 0 or parameters['phono_char_dim'] > 0 or  parameters['word_dim'] > 0
assert 0. <= parameters['dropout'] < 1.0
assert parameters['tag_scheme'] in ['iob', 'iobes']
assert parameters["train_transfer_model"] is False, "Transfer case not implemented yet"
assert not(parameters["ortho_char_dim"] > 0 and parameters["train_transfer_model"]), "Transfer model " \
                    "=> Can't use orthographic features so char dim option ignored. Please set --train_transfer_model " \
                    "to False and restart"
assert parameters["use_convolution"] == False, "Convolution not implemented yet. Please set --use_convolution 0 and restart."

# Check evaluation script / folders
if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)
if not os.path.exists(models_path):
    os.makedirs(models_path)

print("Arg sanity checks done. Instantiating model...")

# Initialize model
model = Model(parameters=parameters, new_model_dir=models_path)
print "Model location: %s" % model.model_path

# Data parameters
lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']

print("Loading sentences")

# Load sentences
train_sentences = loader.load_sentences(opts.train, zeros)
dev_sentences = loader.load_sentences(opts.dev, zeros)
test_sentences = loader.load_sentences(opts.test, zeros)

# loader loads a list of list of lists i.e.
#  a list of sentences, each sentence itself a list of
# [word, . . ., tag]  list

print("Updating tag scheme if needed...")

# Use selected tagging scheme (IOB / IOBES)
update_tag_scheme(train_sentences, tag_scheme)
update_tag_scheme(dev_sentences, tag_scheme)
update_tag_scheme(test_sentences, tag_scheme)


print("Loading CCA vects")
src_lang_wvecs_dict = loader.load_multi_cca_vects(parameters["src_lang_cca_vec_location"])

# update_tag_scheme updates tags (mutable lists changes within function) iob2 or iobes tags

# Reading in CCA vects

# We don't need to create the word and character dictionaries since
# we provide word/char time_distr_dense directly as input to the LSTMs

# # Create a dictionary / mapping of words
# # If we use pretrained time_distr_dense, we add them to the dictionary.
# if parameters['pre_emb']:
#     dico_words_train = word_mapping(train_sentences, lower)[0]
#     # If lower is enabled, by converting to lower, we lose the
#     # difference between "İ" and "I" for example, in addition
#     # to capitalization information. This shouldn't cause
#     # problems for word time_distr_dense but it will for phonotactic
#     # time_distr_dense. Consequently, don't use dico_words.keys()
#     # to get the word vocabulary.
#
#     # word_mapping returns dico, word_to_id, id_to_word
#
#     dico_words, word_to_id, id_to_word = augment_with_pretrained(
#         dico_words_train.copy(),
#         parameters['pre_emb'],
#         list(itertools.chain.from_iterable(
#             [[w[0] for w in s] for s in dev_sentences + test_sentences])
#         ) if not parameters['all_emb'] else None
#     )
#     # augment_with_pretrained simply adds dev/test set words not in train set
#     # to dico_words_copy and returns it along with the new word_to_id and
#     # id_to_word mappings
#
#
# else:
#     dico_words, word_to_id, id_to_word = word_mapping(train_sentences, lower)
#     dico_words_train = dico_words

# # Create a dictionary and a mapping for words / POS tags / tags
# dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
# TO DO:
# Construct id to tags by taking into consideration the dev and test data as well,
# since it is possible to not have training data for a tag
print("Obtaining tag_to_id dict")
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)
# Returns the dictionary of tags to freq, tag_to_id, id_to_tag

# TO DO :
# HANDLE UNKS!! ---> This should be changed in the word vec creation stage
# Since in low data scenario, most tokens wil be singletons, switch tokens
# with unk with a certain probability

print("Preparing data for train/dev/test")
train_data = loader.prepare_phono_dataset(train_sentences, src_lang_wvecs_dict, tag_to_id, src_epi, parameters)
dev_data = loader.prepare_phono_dataset(dev_sentences, src_lang_wvecs_dict, tag_to_id, src_epi, parameters)
test_data = loader.prepare_phono_dataset(test_sentences, src_lang_wvecs_dict, tag_to_id, src_epi, parameters)

# prepare_phono_dataset returns a list of dictionaries, each containing the appropriate data

# TO DO:
# Must implement your own dataset preparation method
# # Index data
# train_data = prepare_dataset(
#     train_sentences, word_to_id, char_to_id, tag_to_id, lower
# )
# dev_data = prepare_dataset(
#     dev_sentences, word_to_id, char_to_id, tag_to_id, lower
# )
# test_data = prepare_dataset(
#     test_sentences, word_to_id, char_to_id, tag_to_id, lower
# )

print "%i / %i / %i sentences in train / dev / test." % (
    len(train_data), len(dev_data), len(test_data))

# Save the mappings to disk
print 'Saving the mappings to disk...'
# model.save_mappings(id_to_word, id_to_char, id_to_tag)
model.save_mappings(id_to_tag)
# Simply saves the id_to_tag

# Build the model
print("Building model")
f_train, f_eval = model.build(**parameters)

# Reload previous model values
if opts.reload:
    print 'Reloading previous model...'
    model.reload()

print("Starting the training procedure")
#
# Train network
#
# n_epochs = 100  # number of epochs over the training set
freq_eval = 1000  # evaluate on dev every freq_eval steps
best_dev = -np.inf
best_test = -np.inf
batch_wise_dev_scores = []
batch_wise_test_scores = []
epoch_wise_dev_scores = []
epoch_wise_test_scores = []
test_for_best_dev = []
count = 0
for epoch in xrange(n_epochs):
    epoch_costs = []
    print "Starting epoch %i..." % epoch
    for i, index in enumerate(np.random.permutation(len(train_data))):
        count += 1
        # input = create_input(train_data[index], parameters, True, singletons)
        input = create_phono_model_input(train_data[index], parameters, True)
        new_cost = f_train(*input)
        epoch_costs.append(new_cost)
        if i % 50 == 0 and i > 0 == 0:
            print "%i, cost average: %f" % (i, np.mean(epoch_costs[-50:]))
        if count % freq_eval == 0:
            dev_score = evaluate_phono_model(parameters, f_eval, dev_sentences,
                                 dev_data, id_to_tag, dico_tags)
            test_score = evaluate_phono_model(parameters, f_eval, test_sentences,
                                  test_data, id_to_tag, dico_tags)
            batch_wise_dev_scores.append(dev_score)
            batch_wise_test_scores.append(test_score)
            print "Score on dev: %.5f" % dev_score
            print "Score on test: %.5f" % test_score
            if dev_score > best_dev:
                best_dev = dev_score
                test_for_best_dev.append(test_score)
                print "New best score on dev."
                print "Saving model to disk..."
                model.save()
            if test_score > best_test:
                best_test = test_score
                print "New best score on test."
    print "Epoch %i done. Average cost: %f" % (epoch, np.mean(epoch_costs))
    epoch_wise_dev_scores.append(dev_score)
    epoch_wise_test_scores.append(test_score)
print("List of all testfor best_dev:")
print(test_for_best_dev)
print("Best possible test score by choosing a new best on dev at some point: %f", max(test_for_best_dev))

cPickle.dump(epoch_wise_dev_scores, open(os.path.join(model.model_path, "epoch_wise_dev_scores"), 'wb'))
print(epoch_wise_dev_scores[-5:])
cPickle.dump(epoch_wise_test_scores, open(os.path.join(model.model_path, "epoch_wise_test_scores"), 'wb'))
print(epoch_wise_test_scores[-5:])
cPickle.dump(batch_wise_dev_scores, open(os.path.join(model.model_path, "batch_wise_dev_scores"), 'wb'))
print(batch_wise_dev_scores[-5:])
cPickle.dump(batch_wise_test_scores, open(os.path.join(model.model_path, "batch_wise_test_scores"), 'wb'))
print(batch_wise_test_scores[-5:])
