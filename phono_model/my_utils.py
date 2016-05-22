# -*- coding: utf-8 -*-
import numpy as np
import pickle
import sys
import codecs

sys.path.append("../epitran/epitran/bin/")
sys.path.append("../epitran/epitran/")


def read_vocab_file(file, vocab_set=set()):
    """
    :param file: The word vocabulary file
    :param vocab_set: a vocabulary set previously produced by this same method
    :return: A set of all unqiue words present in the vocabulary file
    """
    print("Reading file ", file)
    with codecs.open(file, "r", encoding="utf-8") as vocab_file:
        for line in vocab_file:
            word = line.strip(" \t\r\n")
            if word and word not in vocab_set:
                vocab_set.add(word)
    return(vocab_set)

def get_one_hot(one_hot_element_index, vect_dim):
    assert one_hot_element_index < vect_dim, "Index of the one hot element out of vector range"
    ret_vect = np.zeros(vect_dim, np.int)
    ret_vect[one_hot_element_index] = 1
    return(ret_vect)

def get_phono_vecs(word_vocab, epi, word_categories, phono_char_to_id=None, ortho_char_to_id=None):
    phono_vocab = set()
    ortho_vocab = set()

    word_to_phono_mats_dict = {}
    word_to_phono_char_ids_dict = {}
    word_to_ortho_char_ids_dict = {}
    word_to_cats_vecs_dict = {}
    word_to_caps_dict = {}

    for word in word_vocab:
        phono_vecs = epi.word_to_segs(word)
        word_phono_vecs = []
        word_ortho_chars = []
        word_phono_chars = []
        word_cats_vecs = []
        word_caps_vect = []
        for phono_vec in phono_vecs:
            (character_category,
             is_upper,
             orthographic_form,
             phonetic_form,
             in_ipa_punc_space,
             phonological_feature_vector) = phono_vec
            # phono_vocab.add(in_ipa_punc_space)
            phono_vocab.add(phonetic_form)
            ortho_vocab.add(orthographic_form)
            word_phono_vecs.append(phonological_feature_vector)
            word_ortho_chars.append(orthographic_form)
            # word_phono_chars.append(in_ipa_punc_space)
            word_phono_chars.append(phonetic_form)
            cat_vect = []
            for cat in word_categories:
                if character_category == cat:
                    cat_vect.append(1)
                else:
                    cat_vect.append(0)
            word_cats_vecs.append(cat_vect)
            word_caps_vect.append(is_upper)
        word_phono_vecs = np.array(word_phono_vecs)
        word_cats_vecs = np.array(word_cats_vecs)
        word_caps_vect = np.array(word_caps_vect)
        word_to_phono_mats_dict[word] = word_phono_vecs
        word_to_ortho_char_ids_dict[word] = word_ortho_chars
        word_to_phono_char_ids_dict[word] = word_phono_chars
        word_to_cats_vecs_dict[word] = word_cats_vecs
        word_to_caps_dict[word] = word_caps_vect

    # Probably do some rare_thresholding on characters to be able to learn some parameters for unk character
    if phono_char_to_id is None:
        phono_char_to_id = {phono_char: index for index, phono_char in enumerate(phono_vocab)}
    else:
        new_phono_vocab = phono_vocab.difference(phono_char_to_id.keys())
        phono_char_to_id.update({new_phono_char: len(phono_char_to_id) + index for index, new_phono_char in enumerate(new_phono_vocab)})
    assert len(phono_char_to_id.values()) == len(set(phono_char_to_id.values())), "Some phono character indices repeated"
    if ortho_char_to_id is None:
        ortho_char_to_id = {ortho_char: index for index, ortho_char in enumerate(ortho_vocab)}
    else:
        new_ortho_vocab = ortho_vocab.difference(ortho_char_to_id.keys())
        ortho_char_to_id.update(
            {new_ortho_char: len(ortho_char_to_id) + index for index, new_ortho_char in enumerate(new_ortho_vocab)})
    assert len(ortho_char_to_id.values()) == len(set(ortho_char_to_id.values())), "Some ortho character indices repeated"
    # Unk is handled by adding another dimension to the char id vector and setting it in case of an unseen char
    # word_phono_mat_dict = {}
    # word_phono_char_vecs_mat_dict = {}
    # word_ortho_char_vecs_mat_dict = {}
    # word_cats_mat_dict = {}
    # word_caps_vect_dict = {}

    # for word in word_vocab:
    #     word_phono_mat = word_to_phono_mats_dict[word]
    #     word_phono_char_vecs_mat = np.array([get_one_hot(phono_char_to_id[phono_char], len(phono_char_to_id) + 1) for phono_char in  word_to_phono_char_ids_dict[word]])
    #     word_ortho_char_vecs_mat = np.array([get_one_hot(ortho_char_to_id[ortho_char], len(ortho_char_to_id) + 1)  for ortho_char in word_to_ortho_char_ids_dict[word]])
    #     word_cats_mat = word_to_cats_vecs_dict[word]
    #     word_caps_vect = word_to_caps_dict[word]
    #     assert len(word_phono_mat) == len(word_phono_char_vecs_mat) == len(word_ortho_char_vecs_mat) ==\
    #            len(word_cats_mat) == len(word_caps_vect), "Word vector/matrix dimensions don't match"
    #
    #     word_phono_mat_dict[word] = word_phono_mat
    #     word_phono_char_vecs_mat_dict[word] = word_phono_char_vecs_mat
    #     word_ortho_char_vecs_mat_dict[word] = word_ortho_char_vecs_mat
    #     word_cats_mat_dict[word] = word_cats_mat
    #     word_caps_vect_dict[word] = word_caps_vect

    # return(word_phono_mat_dict, word_phono_char_vecs_mat_dict, word_ortho_char_vecs_mat_dict,
    #        word_cats_mat_dict, word_caps_vect_dict, phono_char_to_id, ortho_char_to_id)

    return(word_to_phono_mats_dict,
    word_to_phono_char_ids_dict,
    word_to_ortho_char_ids_dict,
    word_to_cats_vecs_dict,
    word_to_caps_dict,
    phono_char_to_id,
    ortho_char_to_id)

def get_char_vec_mats(chars_vect, char_to_id_dict):
    return(np.array([get_one_hot(char_to_id_dict.get(phono_char, len(char_to_id_dict)), len(char_to_id_dict) + 1) for phono_char in chars_vect]))

def concat_mat_columns_wise(mat1, mat2):
    return(np.concatenate(np.array(mat1), np.array(mat2), axis=1))

def concat_mat_vect_column_wise(mat, vect):
    return(np.concatenate(np.array(mat), np.array(vect)[:, None], axis=1))