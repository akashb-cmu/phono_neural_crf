from my_utils import *
import os
sys.path.append("../epitran/epitran/bin/")
sys.path.append("../epitran/epitran/")
import vector
from copy import deepcopy
import codecs

import argparse

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("-S", "--src_lang",
                        help="Name of the source language",
                        type=str)
arg_parser.add_argument("-SV", "--src_vocab_file_prefix", help="Name prefix for vocabulary files with all source language"
                                                               "words for which features need to be extract",
                        type=str)
arg_parser.add_argument("-SD", "--src_vocab_dir", help="Directory that contains the source language vocabulary files",
                        type=str)
arg_parser.add_argument("-SC", "--src_lang_code", help="Epitran language code for the source language", choices=["aze-Cyrl",
                        "aze-Latn", "deu-Latn", "deu-Latn-np", "fra-Latn", "fra-Latn-np", "hau-Latn",
                        "ind-Latn", "jav-Latn", "kaz-Cyrl", "kaz-Latn", "kir-Arab", "kir-Cyrl", "kir-Latn", "nld-Latn",
                        "spa-Latn", "tuk-Cyrl", "tuk-Latn", "tur-Latn", "yor-Latn", "uig-Arab", "uzb-Cyrl", "uzb-Latn"],
                        default="tur-Latn",
                        type=str)
arg_parser.add_argument("-SS", "--src_lang_space", help="Epitran language space code for the source language",
                        choices=["tur-Latn-suf", "tur-Latn-nosuf", "uzb-Latn-suf", "spa-Latn", "nld-Latn", "deu-Latn"],
                        default="tur-Latn-suf",
                        type=str)

arg_parser.add_argument("-T", "--trg_lang",
                        help="Name of the target language",
                        type=str)
arg_parser.add_argument("-TV", "--trg_vocab_file_prefix",
                        help="Name prefix for vocabulary files with all target language words for which features need "
                             "to be extract",
                        type=str)
arg_parser.add_argument("-TD", "--trg_vocab_dir", help="Directory that contains the target language vocabulary files", type=str)
arg_parser.add_argument("-TC", "--trg_lang_code", help="Epitran language code for the source language",choices=["aze-Cyrl",
                        "aze-Latn", "deu-Latn", "deu-Latn-np", "fra-Latn", "fra-Latn-np", "hau-Latn",
                        "ind-Latn", "jav-Latn", "kaz-Cyrl", "kaz-Latn", "kir-Arab", "kir-Cyrl", "kir-Latn", "nld-Latn",
                        "spa-Latn", "tuk-Cyrl", "tuk-Latn", "tur-Latn", "yor-Latn", "uig-Arab", "uzb-Cyrl", "uzb-Latn"], default="uzb-Latn",
                        type=str)
arg_parser.add_argument("-TS", "--trg_lang_space", help="Epitran language space code for the target language",
                        choices=["tur-Latn-suf", "tur-Latn-nosuf", "uzb-Latn-suf", "spa-Latn", "nld-Latn", "deu-Latn"],
                        default="uzb-Latn-suf",
                        type=str)

args = arg_parser.parse_args()
print("Args used for this run:")
print(args)
src_lang = args.src_lang
src_vocab_file_prefix = args.src_vocab_file_prefix
src_vocab_dir = args.src_vocab_dir
src_lang_code = args.src_lang_code
src_lang_space = args.src_lang_space

trg_lang = args.trg_lang
trg_vocab_file_prefix = args.trg_vocab_file_prefix
trg_vocab_dir = args.trg_vocab_dir
trg_lang_code = args.trg_lang_code
trg_lang_space = args.trg_lang_space

word_categories = ['L', 'M', 'N', 'P', 'S', 'Z', 'C']
# List obtained from epitran repo https://github.com/dmort27/epitran


def read_all_vocabs(dir, file_prefix):
    vocab_set = set()
    for [path, dirs, files] in os.walk(dir):
        for file in files:
            if file_prefix in file:
                vocab_set = read_vocab_file(os.path.join(path, file), vocab_set=vocab_set)
    return(vocab_set)

src_vocab = read_all_vocabs(src_vocab_dir, src_vocab_file_prefix)
trg_vocab = read_all_vocabs(trg_vocab_dir, trg_vocab_file_prefix)

print(len(src_vocab))
print(len(trg_vocab))

src_epi = vector.VectorsWithIPASpace(src_lang_code, src_lang_space)
trg_epi = vector.VectorsWithIPASpace(trg_lang_code, trg_lang_space)


def write_word_vecs(vecs_mat_dict, word, output_file):
    with codecs.open(output_file, mode='a', encoding='utf-8') as op_file:
        op_file.write(word + "\n")
        for vec in vecs_mat_dict:
            vec_str = ""
            type_vec = type(vec)
            is_list = isinstance(vec, list) or isinstance(vec, np.ndarray)
            if is_list:
                vec_str = "|".join([str(elt) for elt in vec])
            else:
                vec_str = str(vec)
            op_file.write(vec_str + "\n")
        op_file.write("\n")

def write_files_to_adhi(file_prefix, word_vocab,
                        word_to_phono_mats_dict, \
                        word_to_phono_char_ids_dict, \
                        word_to_ortho_char_ids_dict, \
                        word_to_cats_vecs_dict, \
                        word_to_caps_dict, \
                        phono_char_to_id, \
                        ortho_char_to_id):
    skipped_words = []
    phono_feats_file = file_prefix + "phono.vecs"
    phono_ids_file = file_prefix + "phono_ids.vecs"
    ortho_ids_file = file_prefix + "ortho_ids.vecs"
    cats_vecs_file = file_prefix + "cats.vecs"
    caps_indicator_file = file_prefix + "caps.vecs"
    for word in word_vocab:
        try:
            word_phono_feats_mat = word_to_phono_mats_dict[word]
            word_phono_char_ids_vect = word_to_phono_char_ids_dict[word]
            word_phono_char_vecs = get_char_vec_mats(word_phono_char_ids_vect, phono_char_to_id)
            word_ortho_char_ids_vect = word_to_ortho_char_ids_dict[word]
            word_ortho_char_vecs = get_char_vec_mats(word_ortho_char_ids_vect, ortho_char_to_id)
            word_cats_vect = word_to_cats_vecs_dict[word]
            word_caps_vect = word_to_caps_dict[word]
        except:
            print("Word %s skipped since not present in some feature dict"%(word))
            skipped_words.append(word)
            continue
        write_word_vecs(word_phono_feats_mat, word, phono_feats_file)
        write_word_vecs(word_phono_char_vecs, word, phono_ids_file)
        write_word_vecs(word_ortho_char_vecs, word, ortho_ids_file)
        write_word_vecs(word_cats_vect, word, cats_vecs_file)
        write_word_vecs(word_caps_vect, word, caps_indicator_file)
    return(skipped_words)




print("Case separate feats for each language")
# Each language uses its own ipa space. No unification is performed

src_word_to_phono_mats_dict, \
src_word_to_phono_char_ids_dict, \
src_word_to_ortho_char_ids_dict, \
src_word_to_cats_vecs_dict, \
src_word_to_caps_dict, \
src_phono_char_to_id, \
src_ortho_char_to_id  = get_phono_vecs(src_vocab, src_epi, word_categories)

trg_word_to_phono_mats_dict, \
trg_word_to_phono_char_ids_dict, \
trg_word_to_ortho_char_ids_dict, \
trg_word_to_cats_vecs_dict, \
trg_word_to_caps_dict, \
trg_phono_char_to_id, \
trg_ortho_char_to_id  = get_phono_vecs(trg_vocab, trg_epi, word_categories)



print("Ensuring NO unification is performed")

# print(set(src_phono_char_to_id).difference(trg_phono_char_to_id))
# print(set(trg_phono_char_to_id).difference(src_phono_char_to_id))
for key in src_phono_char_to_id.keys():
    if key not in trg_phono_char_to_id.keys():
        print(key, " not in trg phono_vocab")
        continue
    if src_phono_char_to_id[key] != trg_phono_char_to_id[key]:
        print(src_phono_char_to_id[key], trg_phono_char_to_id[key])

for key in trg_phono_char_to_id.keys():
    if key not in src_phono_char_to_id.keys():
        print(key, " not in src phono_vocab")
        continue
    if src_phono_char_to_id[key] != trg_phono_char_to_id[key]:
        print(src_phono_char_to_id[key], trg_phono_char_to_id[key])



SRC_IN_SRC_SPACE = src_lang + "_in_" + src_lang +"_ipa_"
TRG_IN_TRG_SPACE = trg_lang + "_in_" + trg_lang +"_ipa_"

write_files_to_adhi(SRC_IN_SRC_SPACE, src_vocab,
                    src_word_to_phono_mats_dict, \
                    src_word_to_phono_char_ids_dict, \
                    src_word_to_ortho_char_ids_dict, \
                    src_word_to_cats_vecs_dict, \
                    src_word_to_caps_dict, \
                    src_phono_char_to_id, \
                    src_ortho_char_to_id
                    )

write_files_to_adhi(TRG_IN_TRG_SPACE, trg_vocab,
                    trg_word_to_phono_mats_dict, \
                    trg_word_to_phono_char_ids_dict, \
                    trg_word_to_ortho_char_ids_dict, \
                    trg_word_to_cats_vecs_dict, \
                    trg_word_to_caps_dict, \
                    trg_phono_char_to_id, \
                    trg_ortho_char_to_id
                    )


# Each language uses its own ipa space and
# ipa space is unified by combining the phonetic character vocabularies
print("Combined vocabulary space case")

src_word_to_phono_mats_dict,\
src_word_to_phono_char_ids_dict,\
src_word_to_ortho_char_ids_dict,\
src_word_to_cats_vecs_dict,\
src_word_to_caps_dict,\
src_phono_char_to_id,\
src_ortho_char_to_id  = get_phono_vecs(src_vocab, src_epi, word_categories)

# src_phono_char_to_id_copy = deepcopy(src_phono_char_to_id)
# with codecs.open("pre_src_dict_delete", "w", encoding="utf-8") as src_dict_file:
#     src_dict_file.write(str(src_phono_char_to_id_copy))

trg_word_to_phono_mats_dict, \
trg_word_to_phono_char_ids_dict, \
trg_word_to_ortho_char_ids_dict, \
trg_word_to_cats_vecs_dict, \
trg_word_to_caps_dict, \
trg_phono_char_to_id, \
trg_ortho_char_to_id  = get_phono_vecs(trg_vocab, trg_epi,
                                       word_categories, phono_char_to_id=src_phono_char_to_id)

print("Verifying unification")

# print(set(src_phono_char_to_id).difference(trg_phono_char_to_id))
# print(set(trg_phono_char_to_id).difference(src_phono_char_to_id))
for key in src_phono_char_to_id.keys():
    if key not in trg_phono_char_to_id.keys():
        print(key, " not in trg phono_vocab")
        continue
    if src_phono_char_to_id[key] != trg_phono_char_to_id[key]:
        print(src_phono_char_to_id[key], trg_phono_char_to_id[key])

for key in trg_phono_char_to_id.keys():
    if key not in src_phono_char_to_id.keys():
        print(key, " not in src phono_vocab")
        continue
    if src_phono_char_to_id[key] != trg_phono_char_to_id[key]:
        print(src_phono_char_to_id[key], trg_phono_char_to_id[key])

print("Vocabs sanity check passed")

# check_val =  src_phono_char_to_id == trg_ortho_char_to_id
# unmatched_item = set(src_phono_char_to_id.items()) ^ set(trg_ortho_char_to_id.items())

# with codecs.open("trg_dict_delete", "w", encoding="utf-8") as trg_dict_file:
#     trg_dict_file.write(str(trg_phono_char_to_id))
#
#
# print(len(unmatched_item))
# print(unmatched_item)
# print(check_val)

# raw_input("Enter to continue!")
# assert src_phono_char_to_id == trg_ortho_char_to_id, "Vocabs not unified"

SRC_IN_COMBO_SPACE = src_lang + "_in_combo_unified_ipa_"
TRG_IN_COMBO_SPACE = trg_lang + "_in_combo_unified_ipa_"

write_files_to_adhi(SRC_IN_COMBO_SPACE, src_vocab,
                    src_word_to_phono_mats_dict, \
                    src_word_to_phono_char_ids_dict, \
                    src_word_to_ortho_char_ids_dict, \
                    src_word_to_cats_vecs_dict, \
                    src_word_to_caps_dict, \
                    src_phono_char_to_id, \
                    src_ortho_char_to_id
                    )

write_files_to_adhi(TRG_IN_COMBO_SPACE, trg_vocab,
                    trg_word_to_phono_mats_dict, \
                    trg_word_to_phono_char_ids_dict, \
                    trg_word_to_ortho_char_ids_dict, \
                    trg_word_to_cats_vecs_dict, \
                    trg_word_to_caps_dict, \
                    trg_phono_char_to_id, \
                    trg_ortho_char_to_id
                    )


# Case where src is the trg language's ipa space
print("Case of src projected into trg language space")

src_word_to_phono_mats_dict, \
src_word_to_phono_char_ids_dict, \
src_word_to_ortho_char_ids_dict, \
src_word_to_cats_vecs_dict, \
src_word_to_caps_dict, \
src_phono_char_to_id, \
src_ortho_char_to_id  = get_phono_vecs(src_vocab, trg_epi, word_categories)

trg_word_to_phono_mats_dict, \
trg_word_to_phono_char_ids_dict, \
trg_word_to_ortho_char_ids_dict, \
trg_word_to_cats_vecs_dict, \
trg_word_to_caps_dict, \
trg_phono_char_to_id, \
trg_ortho_char_to_id  = get_phono_vecs(trg_vocab, trg_epi,
                                       word_categories, phono_char_to_id=src_phono_char_to_id)

print("Verifying unification")

# print(set(src_phono_char_to_id).difference(trg_phono_char_to_id))
# print(set(trg_phono_char_to_id).difference(src_phono_char_to_id))
for key in src_phono_char_to_id.keys():
    if key not in trg_phono_char_to_id.keys():
        print(key, " not in trg phono_vocab")
        continue
    if src_phono_char_to_id[key] != trg_phono_char_to_id[key]:
        print(src_phono_char_to_id[key], trg_phono_char_to_id[key])

for key in trg_phono_char_to_id.keys():
    if key not in src_phono_char_to_id.keys():
        print(key, " not in src phono_vocab")
        continue
    if src_phono_char_to_id[key] != trg_phono_char_to_id[key]:
        print(src_phono_char_to_id[key], trg_phono_char_to_id[key])

print("Vocabs sanity check passed")

SRC_IN_TRG_SPACE = src_lang + "_in_" + trg_lang  +"_unified_ipa_"
TRG_IN_TRG_SPACE = trg_lang + "_in_" + trg_lang  + "_unified_ipa_"

write_files_to_adhi(SRC_IN_TRG_SPACE, src_vocab,
                    src_word_to_phono_mats_dict, \
                    src_word_to_phono_char_ids_dict, \
                    src_word_to_ortho_char_ids_dict, \
                    src_word_to_cats_vecs_dict, \
                    src_word_to_caps_dict, \
                    src_phono_char_to_id, \
                    src_ortho_char_to_id
                    )

write_files_to_adhi(TRG_IN_TRG_SPACE, trg_vocab,
                    trg_word_to_phono_mats_dict, \
                    trg_word_to_phono_char_ids_dict, \
                    trg_word_to_ortho_char_ids_dict, \
                    trg_word_to_cats_vecs_dict, \
                    trg_word_to_caps_dict, \
                    trg_phono_char_to_id, \
                    trg_ortho_char_to_id
                    )

# Case where trg is src language's ipa space
print("Case of trg in src space")

src_word_to_phono_mats_dict, \
src_word_to_phono_char_ids_dict, \
src_word_to_ortho_char_ids_dict, \
src_word_to_cats_vecs_dict, \
src_word_to_caps_dict, \
src_phono_char_to_id, \
src_ortho_char_to_id  = get_phono_vecs(src_vocab, src_epi, word_categories)

trg_word_to_phono_mats_dict, \
trg_word_to_phono_char_ids_dict, \
trg_word_to_ortho_char_ids_dict, \
trg_word_to_cats_vecs_dict, \
trg_word_to_caps_dict, \
trg_phono_char_to_id, \
trg_ortho_char_to_id  = get_phono_vecs(trg_vocab, src_epi,
                                       word_categories, phono_char_to_id=src_phono_char_to_id)

print("Verifying unification")

# print(set(src_phono_char_to_id).difference(trg_phono_char_to_id))
# print(set(trg_phono_char_to_id).difference(src_phono_char_to_id))
for key in src_phono_char_to_id.keys():
    if key not in trg_phono_char_to_id.keys():
        print(key, " not in trg phono_vocab")
        continue
    if src_phono_char_to_id[key] != trg_phono_char_to_id[key]:
        print(src_phono_char_to_id[key], trg_phono_char_to_id[key])

for key in trg_phono_char_to_id.keys():
    if key not in src_phono_char_to_id.keys():
        print(key, " not in src phono_vocab")
        continue
    if src_phono_char_to_id[key] != trg_phono_char_to_id[key]:
        print(src_phono_char_to_id[key], trg_phono_char_to_id[key])

print("Vocabs sanity check passed")

SRC_IN_SRC_SPACE = src_lang + "_in_" + src_lang  + "_unified_ipa_"
TRG_IN_SRC_SPACE = trg_lang + "_in_" + src_lang  + "unified_ipa_"

write_files_to_adhi(SRC_IN_SRC_SPACE, src_vocab,
                    src_word_to_phono_mats_dict, \
                    src_word_to_phono_char_ids_dict, \
                    src_word_to_ortho_char_ids_dict, \
                    src_word_to_cats_vecs_dict, \
                    src_word_to_caps_dict, \
                    src_phono_char_to_id, \
                    src_ortho_char_to_id
                    )

write_files_to_adhi(TRG_IN_SRC_SPACE, trg_vocab,
                    trg_word_to_phono_mats_dict, \
                    trg_word_to_phono_char_ids_dict, \
                    trg_word_to_ortho_char_ids_dict, \
                    trg_word_to_cats_vecs_dict, \
                    trg_word_to_caps_dict, \
                    trg_phono_char_to_id, \
                    trg_ortho_char_to_id
                    )

raw_input("Enter to end")