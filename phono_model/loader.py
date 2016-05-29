# -*- coding: utf-8 -*-
import os
import sys
import re
import codecs
from utils import create_dico, create_mapping, zero_digits
from utils import iob2, iob_iobes
# sys.path.append("./phono_model/")
import my_utils
import numpy as np

def load_sentences(path, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.

    Returns a list of list of lists:
    [
        [
            [sent1_word1, . . . , sent1_tag1]
            .
            .
            .
            [sentn_wordn, . . . , sentn_tagn]
        ]
        .
        .
        .
        [
            [sentl_word1, . . . , sentl_tag1]
            .
            .
            .
            [sentl_wordn, . . . , sentl_tagn]
        ]
    ]
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags): # iob2() converts the tags to iob2 if in iob1
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000
    word_to_id, id_to_word = create_mapping(dico)
    print "Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    )
    return dico, word_to_id, id_to_word


def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dico(chars)
    char_to_id, id_to_char = create_mapping(dico)
    print "Found %i unique characters" % len(dico)
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print "Found %i unique named entity tags" % len(dico)
    return dico, tag_to_id, id_to_tag


def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3


def prepare_sentence(str_words, word_to_id, char_to_id, lower=False):
    """
    Prepare a sentence for evaluation.
    """
    def f(x): return x.lower() if lower else x
    words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
             for w in str_words]
    chars = [[char_to_id[c] for c in w if c in char_to_id]
             for w in str_words]
    caps = [cap_feature(w) for w in str_words]
    return {
        'str_words': str_words,
        'words': words,
        'chars': chars,
        'caps': caps
    }


def prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id, lower=False):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    def f(x): return x.lower() if lower else x
    data = []
    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                 for w in str_words]
        # Skip characters that are not in the training set
        chars = [[char_to_id[c] for c in w if c in char_to_id]
                 for w in str_words]
        caps = [cap_feature(w) for w in str_words]
        tags = [tag_to_id[w[-1]] for w in s]
        data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'caps': caps,
            'tags': tags,
        })
    return data

def augment_with_pretrained(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print 'Loading pretrained time_distr_dense from %s...' % ext_emb_path
    assert os.path.isfile(ext_emb_path)

    # Load pretrained time_distr_dense from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])
    # pretrained is the set of all  words for which pre-trained time_distr_dense
    # are available

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower()) # word with digits
                # substituted with 0
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word









"""
Phono model functions
"""

def load_multi_cca_vects(vec_file_path):
    w_vecs = {}
    with codecs.open(vec_file_path, "r", encoding="utf-8") as vec_file:
        for line in vec_file:
            line = line.strip(" \t\r\n")
            if len(line) > 0:
                [lang_code, word_vec] = line.split(u":", 1)
                [word, vec] = word_vec.split(" ", 1)
                vec = [float(elt) for elt in vec.split(" ")]
                w_vecs[word] = vec
    return(w_vecs)

def read_epitran_feats(file_path):
    word_to_feats_dict = {}
    with codecs.open(file_path, "r", encoding="utf-8") as ip_file:
        instance = []
        for line in ip_file:
            line = line.strip(" \t\r\n")
            if len(line) > 0:
                instance.append(line)
            else:
                if len(instance) > 1:
                    word = instance[0]
                    word_feat_mat = []
                    for vec_str in instance[1:]:
                        vec = [int(elt) for elt in vec_str.split("|")]
                        word_feat_mat.append(vec)
                    word_to_feats_dict[word] = np.array(word_feat_mat, dtype=np.float32)
                    instance = []
                else:
                    if len(instance) == 1:
                        print("singleton word case")
                    continue
    return(word_to_feats_dict)

def load_epitran_feats(sentences, epi, epitran_feats_file_prefix=None):
    if epitran_feats_file_prefix is None:
        word_vocab = set()
        for s in sentences:
            s_words = [w[0] for w in s]
            word_vocab.update(s_words)
        # TO DO :
        # Ensure the input matrices are float64
        [word_to_phono_mats_dict,
         word_to_phono_char_ids_dict,
         word_to_ortho_char_ids_dict,
         word_to_cats_vecs_dict,
         word_to_caps_dict,
         phono_char_to_id,
         ortho_char_to_id] = my_utils.get_phono_vecs(word_vocab=word_vocab,
                                                    epi=epi,
                                                    word_categories=my_utils.default_word_categories)
    else:
        phono_feats_file = epitran_feats_file_prefix + "phono.vecs"
        phono_ids_file = epitran_feats_file_prefix + "phono_ids.vecs"
        ortho_ids_file = epitran_feats_file_prefix + "ortho_ids.vecs"
        cats_vecs_file = epitran_feats_file_prefix + "cats.vecs"
        caps_indicator_file = epitran_feats_file_prefix + "caps.vecs"
        word_to_phono_mats_dict = read_epitran_feats(phono_feats_file)
        word_to_phono_char_ids_dict = read_epitran_feats(phono_ids_file)
        word_to_ortho_char_ids_dict = read_epitran_feats(ortho_ids_file)
        word_to_cats_vecs_dict = read_epitran_feats(cats_vecs_file)
        word_to_caps_dict = read_epitran_feats(caps_indicator_file)

    return([word_to_phono_mats_dict,
            word_to_phono_char_ids_dict,
            word_to_ortho_char_ids_dict,
            word_to_cats_vecs_dict,
            word_to_caps_dict])



def prepare_phono_dataset(sentences, word_vec_dict, tag_to_id, epi, parameters):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - ortho_char_vecs (both forward and backward)
        - w_ortho_len for each word in ortho form
        - phono_char_vecs (both forward and backward)
        - w_phono_len for each word in phono form
        - tag indexes

        No sparse token and type
    """
    # print(word_vec_dict[u'Bugün'])
    assert parameters["use_type_sparse_feats"] == False, "Type sparse feats not yet implemented"
    assert parameters["use_token_sparse_feats"] == False, "Token sparse feats not yet implemented"
    parameters["token_sparse_feats_input_dim"] = 0
    parameters["type_sparse_feats_input_dim"] = 0
    lower = parameters.get("lower", False)
    [word_to_phono_mats_dict,
     word_to_phono_char_ids_dict,
     word_to_ortho_char_ids_dict,
     word_to_cats_vecs_dict,
     word_to_caps_dict] = load_epitran_feats(sentences, epi,
                                             epitran_feats_file_prefix=parameters["src_lang_epi_vec_files_path_prefix"])


    def f(x): return x.lower() if lower else x
    data = []
    for s in sentences:
        instance = {}
        str_words = [w[0] for w in s]
        tags = np.array([tag_to_id[w[-1]] for w in s], dtype=np.int32)
        instance["tag_ids"] = tags
        if parameters['word_dim'] > 0:
            try:
                word_vecs = np.array([word_vec_dict[f(w)] for w in str_words], dtype=np.float32)
            except:
                print("Problem in case of ", str_words)
            assert len(word_vecs.shape) == 2, "Word_vecs not of appropriate shape"
            instance["word_vecs"] = word_vecs
            parameters["word_vec_input_dim"] = word_vecs.shape[1]
        else:
            parameters["word_vec_input_dim"] = 0
        if parameters["phono_char_dim"] > 0:
            phono_mats = []
            rev_phono_mats = []
            # max_wchar_len = max([word_to_phono_mats_dict[w].shape[0] for w in str_words])
            try:
                # phono_char_pos = np.array([word_to_phono_mats_dict[w].shape[0] for w in str_words], dtype=np.int32)
                phono_char_pos = np.array([word_to_phono_mats_dict[w].shape[0] - 1 for w in str_words], dtype=np.int32)
            except:
                print("Problem in case of ", str_words)
                raw_input("Enter to continue")
            max_wchar_len = max(phono_char_pos) + 1
            for word in str_words:
                word_phono_mat = word_to_phono_mats_dict[word]
                word_phono_id_mat = word_to_phono_char_ids_dict[word]
                word_caps_vec = word_to_caps_dict[word]
                word_cats_vecs = word_to_cats_vecs_dict[word]
                combo_mat = np.concatenate([word_phono_mat, word_phono_id_mat] + ([word_caps_vec] if parameters["use_caps"] else [])
                                           + ([word_cats_vecs] if parameters["use_cats"] else []), axis=1)
                rev_combo_mat = combo_mat[::-1, :]
                pw_len = combo_mat.shape[0]
                combo_mat = np.pad(combo_mat, pad_width=((0, max_wchar_len - pw_len), (0, 0)), mode="constant", constant_values=0)
                rev_combo_mat = np.pad(rev_combo_mat, pad_width=((0, max_wchar_len - pw_len), (0, 0)), mode="constant",
                                   constant_values=0)
                phono_mats.append(combo_mat)
                rev_phono_mats.append(rev_combo_mat)
            phono_mats = np.array(phono_mats, dtype=np.int32)
            rev_phono_mats = np.array(rev_phono_mats, dtype=np.int32)
            assert len(phono_mats.shape) == 3 and phono_mats.shape[1] == max_wchar_len, "Phono mats is not of appropriate shape"
            assert len(rev_phono_mats.shape) == 3 and rev_phono_mats.shape[1] == max_wchar_len, \
                "Rev Phono mats is not of appropriate shape"
            instance["phono_char_for_vecs"] = phono_mats
            instance["phono_char_rev_vecs"] = rev_phono_mats
            instance["phono_char_pos_ids"] = phono_char_pos
            parameters["phono_char_input_dim"] = phono_mats.shape[2]
        else:
            parameters["phono_char_input_dim"] = 0
        if parameters["ortho_char_dim"] > 0:
            ortho_mats = []
            rev_ortho_mats = []
            # max_wchar_len = max([word_to_ortho_char_ids_dict[w].shape[0] for w in str_words])
            # ortho_char_pos = np.array([word_to_ortho_char_ids_dict[w].shape[0] for w in str_words], dtype=np.int32)
            ortho_char_pos = np.array([word_to_ortho_char_ids_dict[w].shape[0] - 1 for w in str_words], dtype=np.int32)
            max_wchar_len = max(ortho_char_pos) + 1
            for word in str_words:
                word_ortho_ids_mat = word_to_ortho_char_ids_dict[word]
                word_caps_vec = word_to_caps_dict[word]
                word_cats_vecs = word_to_cats_vecs_dict[word]
                try:
                    word_ortho_ids_mat = np.concatenate([word_ortho_ids_mat] + ([word_caps_vec] if parameters["use_caps"] else [])
                                                    + ([word_cats_vecs] if parameters["use_cats"] else []), axis=1)
                except:
                    print("concat problem")
                    raw_input("Enter to continue")
                rev_word_ortho_ids_mat = word_ortho_ids_mat[::-1, :]
                pw_len = word_ortho_ids_mat.shape[0]
                word_ortho_ids_mat = np.pad(word_ortho_ids_mat, pad_width=((0, max_wchar_len - pw_len), (0, 0)), mode="constant",
                                   constant_values=0)
                rev_word_ortho_ids_mat = np.pad(rev_word_ortho_ids_mat, pad_width=((0, max_wchar_len - pw_len), (0, 0)), mode="constant",
                                       constant_values=0)
                ortho_mats.append(word_ortho_ids_mat)
                rev_ortho_mats.append(rev_word_ortho_ids_mat)
            ortho_mats = np.array(ortho_mats, dtype=np.int32)
            rev_ortho_mats = np.array(rev_ortho_mats, dtype=np.int32)
            assert len(ortho_mats.shape) == 3 and ortho_mats.shape[
                                                      1] == max_wchar_len, "Ortho mats is not of appropriate shape"
            assert len(rev_ortho_mats.shape) == 3 and rev_ortho_mats.shape[1] == max_wchar_len, \
                "Rev ortho mats is not of appropriate shape"
            instance["ortho_char_for_vecs"] = ortho_mats
            instance["ortho_char_rev_vecs"] = rev_ortho_mats
            instance["ortho_char_pos_ids"] = ortho_char_pos
            parameters["ortho_char_input_dim"] = ortho_mats.shape[2]
        else:
            parameters["ortho_char_input_dim"] = 0



        data.append(instance)
    return data
