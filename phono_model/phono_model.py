import os
import re
import numpy as np
import scipy.io
import theano
import theano.tensor as T
import codecs
import cPickle

from utils import shared, set_values, get_name
from nn import HiddenLayer, EmbeddingLayer, DropoutLayer, LSTM, forward, TimeDistributedDenseLayer
from optimization import Optimization


class Model(object):
    """
    Network architecture.
    """
    def __init__(self, parameters=None, new_model_dir=None, old_model_dir=None):
        """
        Initialize the model. We either provide the parameters and a path where
        we store the models, or the location of a trained model.
        """
        if old_model_dir is None:
            assert parameters and new_model_dir
            # Create a name based on the parameters
            self.parameters = parameters
            self.name = get_name(parameters)
            # Model location
            old_model_dir = os.path.join(new_model_dir, self.name)
            self.model_path = old_model_dir
            self.parameters_path = os.path.join(old_model_dir, 'parameters.pkl')
            self.parameters_txt_path = os.path.join(old_model_dir, 'parameters_readable.txt')
            self.mappings_path = os.path.join(old_model_dir, 'mappings.pkl')
            # Create directory for the model if it does not exist
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            # Save the parameters to disk
            with open(self.parameters_path, 'wb') as f:
                self.parameters = cPickle.dump(parameters, f)
            with open(self.parameters_txt_path, 'wb') as f:
                f.write(str(self.parameters))
        else:
            assert parameters is None and new_model_dir is None
            # Model location
            self.model_path = old_model_dir
            self.parameters_path = os.path.join(old_model_dir, 'parameters.pkl')
            self.mappings_path = os.path.join(old_model_dir, 'mappings.pkl')
            # Load the parameters and the mappings from disk
            with open(self.parameters_path, 'rb') as f:
                self.parameters = cPickle.load(f)
            self.reload_mappings()
        self.components = {}

    # def save_mappings(self, id_to_word, id_to_char, id_to_tag):
    #     """
    #     We need to save the mappings if we want to use the model later.
    #     """
    #     self.id_to_word = id_to_word
    #     self.id_to_char = id_to_char
    #     self.id_to_tag = id_to_tag
    #     with open(self.mappings_path, 'wb') as f:
    #         mappings = {
    #             'id_to_word': self.id_to_word,
    #             'id_to_char': self.id_to_char,
    #             'id_to_tag': self.id_to_tag,
    #         }
    #         cPickle.dump(mappings, f)

    def save_mappings(self, id_to_tag):
        """
        We need to save the mappings if we want to use the model later.
        """
        self.id_to_tag = id_to_tag
        with open(self.mappings_path, 'wb') as f:
            mappings = {
                'id_to_tag': self.id_to_tag,
            }
            cPickle.dump(mappings, f)

    # def reload_mappings(self):
    #     """
    #     Load mappings from disk.
    #     """
    #     with open(self.mappings_path, 'rb') as f:
    #         mappings = cPickle.load(f)
    #     self.id_to_word = mappings['id_to_word']
    #     self.id_to_char = mappings['id_to_char']
    #     self.id_to_tag = mappings['id_to_tag']

    def reload_mappings(self):
        """
        Load mappings from disk.
        """
        with open(self.mappings_path, 'rb') as f:
            mappings = cPickle.load(f)
        self.id_to_tag = mappings['id_to_tag']

    def add_component(self, param):
        """
        Add a new parameter to the network.
        """
        if param.name in self.components:
            raise Exception('The network already has a parameter "%s"!'
                            % param.name)
        self.components[param.name] = param

    def save(self):
        """
        Write components values to disk.
        """
        for name, param in self.components.items():
            param_path = os.path.join(self.model_path, "%s.mat" % name)
            if hasattr(param, 'params'):
                param_values = {p.name: p.get_value() for p in param.params}
            else:
                param_values = {name: param.get_value()}
            scipy.io.savemat(param_path, param_values)

    def reload(self):
        """
        Load components values from disk.
        """
        for name, param in self.components.items():
            param_path = os.path.join(self.model_path, "%s.mat" % name)
            param_values = scipy.io.loadmat(param_path)
            if hasattr(param, 'params'):
                for p in param.params:
                    set_values(p.name, p, param_values[p.name])
            else:
                set_values(name, param, param_values[name])


    # TO DO :
    # Adding priors to the attention vectors
    def TDAttention(self, vec_t, mat_t, col_selector_t):
        # vec_t is a particular vector of size (vect_dim) on which we condition to obtain the attention vector
        # mat_t is the matrix of source vectors of size (n_vects, vect_dim) over which attention is done
        # col_selector_t has the index of the last non-padding column in mat_t
        # trunc_mat = mat_t[:col_selector_t]
        trunc_mat = mat_t[:col_selector_t + 1]
        att_coeffs = T.nnet.softmax(T.dot(vec_t, trunc_mat.T))
        att_vect = T.sum(att_coeffs * trunc_mat.T, axis=1)
        return (att_vect)

    def get_TDAttention_vector(self, conditioning_vecs, source_mats, mat_col_selectors):
        # conditioning vecs is a matrix of vectors of size (seq_len, vect_dim), each of which on which we condition to
        # obtain the attention vector
        #
        # source_mats is the tensor of matrices of size (seq_len, max_char_len, vect_dim) of source vectors
        # over which attention is done
        #
        # mat_col_selectors is an integer vector of size (seq_len, ) which has the indices of the last non-padding
        # column in mat_t for each word in seq
        #
        # Returns the scan result of size (seq_len, vect_dim) i.e. an attention vector for each word in the seqence
        # scan_result, scan_updates = theano.scan(fn=self.TDAttention, outputs_info=None,
        #                                         sequences=[conditioning_vecs, source_mats, mat_col_selectors])
        scan_result, scan_updates = theano.scan(fn=self.TDAttention,
                                                 sequences=[conditioning_vecs, source_mats, mat_col_selectors])

        return(scan_result)

    # TO DO
    def build(self,
              dropout,
              ortho_char_input_dim, # Should be inferred from the input
              ortho_char_dim,
              ortho_char_lstm_dim,
              char_bidirect,
              word_vec_input_dim, # Should be inferred from the input wvecs
              word_dim,  # The vector size after projection of the input vector
              word_lstm_dim,
              word_bidirect,
              lr_method,
              crf,
              use_type_sparse_feats,
              type_sparse_feats_input_dim,  # Can be inferred from the output of the feature extractors
              type_sparse_feats_proj_dim,  # This is a hyper-parameter
              use_token_sparse_feats,
              token_sparse_feats_input_dim,  # Can be inferred from the output of the feature extractors
              # token_sparse_feats_proj_dim,  # This is a hyper-parameter
              use_ortho_attention,
              use_phono_attention,
              # use_convolution,
              phono_char_input_dim, # Can be inferred
              phono_char_dim,
              phono_char_lstm_dim,
              training=True,
              **kwargs
              ):
        """
        Build the network.
        """
        assert word_dim or phono_char_dim or ortho_char_dim, "No input selected while building the network!"
        # Training parameters
        n_tags = len(self.id_to_tag)

        # Network variables
        is_train = T.iscalar('is_train')
        word_vecs = T.dmatrix(name="word_vecs") # A vector for each word in the sentence
                                                #  => matrix: (len_sent, w_emb_dim)
        ortho_char_for_vecs = T.dtensor3(name="ortho_char_for_vecs") # For each char of each word in the sentence, a char vector
        # ortho_char_for_vecs = T.ftensor3(name="ortho_char_for_vecs")
        # => tensor of form: (len_sent, max_wchar_len, char_emb_dim)
        ortho_char_rev_vecs = T.dtensor3(name="ortho_char_rev_vecs")
        # ortho_char_rev_vecs = T.ftensor3(name="ortho_char_rev_vecs")
        # For each char of each word in the sentence, a char vector
        # => tensor of form: (len_sent, max_wchar_len, char_emb_dim)
        phono_char_for_vecs = T.dtensor3(name="phono_char_for_vecs")
        # phono_char_for_vecs = T.ftensor3(name="phono_char_for_vecs")
        # For each char of each word in the sentence, a char vector
        # => tensor of form: (len_sent, max_ortho_char_len, char_emb_dim)
        phono_char_rev_vecs = T.dtensor3(name="phono_char_rev_vecs")
        # phono_char_rev_vecs = T.ftensor3(name="phono_char_rev_vecs")
        # For each char of each word in the sentence, a char vector
        # => tensor of form: (len_sent, max_phono_char_len, char_emb_dim)
        ortho_char_pos_ids = T.ivector(name='ortho_char_pos_ids')
        # The word len for each word in the sentence => vect of form: (len_sent,)
        phono_char_pos_ids = T.ivector(name='phono_char_pos_ids')
        # The word len for each word in the sentence => vect of form: (len_sent,)
        type_sparse_feats = T.imatrix(name="type_sparse_feats")
        # Type sparse features are appended to the input to the word lstm
        # For each word, a vector of type level sparse feats => mat of form: (len_sent, type_sparse_dim)
        token_sparse_feats = T.imatrix(name="token_sparse_feats")
        # Token sparse features are appended to the pre-crf layer
        # For each word, a vector of token level sparse feats => mat of form: (len_sent, token_sparse_dim)

        tag_ids = T.ivector(name='tag_ids')
        # The tag id for each word in the sentence => vect of form: (len_sent,)


        # Sentence length
        s_len = (word_vecs if word_dim else ortho_char_pos_ids if ortho_char_dim else phono_char_pos_ids).shape[0]

        # Final input (all word features)
        input_dim = 0
        inputs = []

        #
        # Word inputs
        #
        if word_dim:
            input_dim += word_dim
            word_layer = HiddenLayer(word_vec_input_dim, word_dim, activation="tanh", name="word_emb_proj")
            # TO DO : Try not using the bias term in the hidden layer
            word_input = word_layer.link(word_vecs)
            inputs.append(word_input)

        #
        # Chars inputs
        #
        if ortho_char_dim:
            input_dim += ortho_char_lstm_dim
            ortho_char_layer = HiddenLayer(ortho_char_input_dim, ortho_char_dim,
                                            activation="tanh", name="ortho_char_emb_proj")
            # TO DO : Try not using bias in the hidden layer
            ortho_char_lstm_for = LSTM(ortho_char_dim, ortho_char_lstm_dim, with_batch=True,
                                       name='ortho_char_lstm_for')
            ortho_char_lstm_rev = LSTM(ortho_char_dim, ortho_char_lstm_dim, with_batch=True,
                                       name='ortho_char_lstm_rev')
            ortho_char_lstm_for.link(ortho_char_layer.link(ortho_char_for_vecs))
            ortho_char_lstm_rev.link(ortho_char_layer.link(ortho_char_rev_vecs))

            ortho_char_for_output = ortho_char_lstm_for.h.dimshuffle((1, 0, 2))[
                T.arange(s_len), ortho_char_pos_ids
            ]
            ortho_char_rev_output = ortho_char_lstm_rev.h.dimshuffle((1, 0, 2))[
                T.arange(s_len), ortho_char_pos_ids
            ]

            inputs.append(ortho_char_for_output)
            if char_bidirect:
                inputs.append(ortho_char_rev_output)
                input_dim += ortho_char_lstm_dim


        if phono_char_dim:
            input_dim += phono_char_lstm_dim
            phono_char_layer = HiddenLayer(phono_char_input_dim, phono_char_dim, activation="tanh",
                                                         name="phono_char_emb_proj")
            # TO DO : Try not using bias in the hidden layer
            phono_char_lstm_for = LSTM(phono_char_dim, phono_char_lstm_dim, with_batch=True,
                                       name='phono_char_lstm_for')
            phono_char_lstm_rev = LSTM(phono_char_dim, phono_char_lstm_dim, with_batch=True,
                                       name='phono_char_lstm_rev')

            phono_char_lstm_for.link(phono_char_layer.link(phono_char_for_vecs))
            phono_char_lstm_rev.link(phono_char_layer.link(phono_char_rev_vecs))

            phono_char_for_output = phono_char_lstm_for.h.dimshuffle((1, 0, 2))[
                T.arange(s_len), phono_char_pos_ids
            ]
            phono_char_rev_output = phono_char_lstm_rev.h.dimshuffle((1, 0, 2))[
                T.arange(s_len), phono_char_pos_ids
            ]

            inputs.append(phono_char_for_output)
            if char_bidirect:
                inputs.append(phono_char_rev_output)
                input_dim += phono_char_lstm_dim

        # Type level sparse feats
        #
        if use_type_sparse_feats:
            input_dim += type_sparse_feats_input_dim
            type_level_sparse_layer = HiddenLayer(type_sparse_feats_input_dim, type_sparse_feats_proj_dim, activation="tanh",
                                                  name='type_level_sparse_layer')
            # TO DO : Try not using the hidden layer here
            inputs.append(type_level_sparse_layer.link(type_sparse_feats))

        # Prepare final input
        if len(inputs) != 1:
            inputs = T.concatenate(inputs, axis=1)
            # TO DO : If using type sparse features, then apply hidden layer after concatenating all inputs
        else:
            inputs = inputs[0]
        #
        # Dropout on final input
        #
        if dropout:
            dropout_layer = DropoutLayer(p=dropout)
            input_train = dropout_layer.link(inputs)
            input_test = (1 - dropout) * inputs
            """
            Drop out involves sampling a vector of bernoulli random variables with a parameter 1-p and using it as a mask
            So, the expected value of the dropped out input is p * (0*x) + (1-p) * (1*x) = (1-p) * x. Since biases will
            on average respond to the expected input value, at test time we multiply test inputs (1-p) to supply the
            expected test input instead.
            """
            inputs = T.switch(T.neq(is_train, 0), input_train, input_test)

        # LSTM for words
        word_lstm_for = LSTM(input_dim, word_lstm_dim, with_batch=False,
                             name='word_lstm_for')
        word_lstm_rev = LSTM(input_dim, word_lstm_dim, with_batch=False,
                             name='word_lstm_rev')
        word_lstm_for.link(inputs)
        word_lstm_rev.link(inputs[::-1, :])
        word_for_output = word_lstm_for.h
        word_rev_output = word_lstm_rev.h[::-1, :]
        lstm_outputs = [word_for_output]
        post_word_lstm_output_size = word_lstm_dim
        if use_token_sparse_feats:
            # token_level_sparse_layer = HiddenLayer(token_sparse_feats_input_dim, token_sparse_feats_proj_dim,
            #                                       activation="tanh",
            #                                       name='token_level_sparse_layer')
            # # TO DO : Try not using the hidden layer here
            # lstm_outputs.append(token_level_sparse_layer.link(token_sparse_feats))
            # post_word_lstm_output_size += token_sparse_feats_proj_dim
            lstm_outputs.append(token_sparse_feats)
            post_word_lstm_output_size += token_sparse_feats_input_dim
        if word_bidirect:
            lstm_outputs.append(word_rev_output)
            post_word_lstm_output_size += word_lstm_dim

        if len(lstm_outputs) > 1:
            final_output = T.concatenate(lstm_outputs, axis=1)
            tanh_layer = HiddenLayer(post_word_lstm_output_size, word_lstm_dim,
                                     name='tanh_layer', activation='tanh')
            final_output = tanh_layer.link(final_output)

        else:
            final_output = word_for_output

        final_pre_crf_input_size = word_lstm_dim
        attention_vectors = []
        attention_vector_size = 0
        if use_ortho_attention and ortho_char_dim:
            # final_ortho_attention_input_layer = HiddenLayer(post_word_lstm_output_size, ortho_char_lstm_dim,
            #                                   name='final_ortho_attention_input_layer', activation='tanh')
            final_ortho_attention_input_layer = HiddenLayer(word_lstm_dim, ortho_char_lstm_dim,
                                                            name='final_ortho_attention_input_layer', activation='tanh')
            final_ortho_attention_input = final_ortho_attention_input_layer.link(final_output)
            # Evaluating attentional vector using a linear projection from final_output since the attention vector
            # must be conditioned on it and dimension must match the char lstm hidden dim.
            ortho_for_attention = self.get_TDAttention_vector(final_ortho_attention_input,
                                                          ortho_char_lstm_for.h.dimshuffle((1, 0, 2)),
                                                          ortho_char_pos_ids)
            if char_bidirect:
                ortho_rev_attention = self.get_TDAttention_vector(final_ortho_attention_input,
                                                                  ortho_char_lstm_rev.h.dimshuffle((1, 0, 2)),
                                                                  ortho_char_pos_ids)
                attention_vectors.append(ortho_rev_attention)
                attention_vector_size += ortho_char_lstm_dim
            attention_vectors.append(ortho_for_attention)
            attention_vector_size += ortho_char_lstm_dim
        if use_phono_attention and phono_char_dim:
            # final_phono_attention_input_layer = HiddenLayer(post_word_lstm_output_size, phono_char_lstm_dim,
            #                                               name='final_phono_attention_input_layer', activation='tanh')
            final_phono_attention_input_layer = HiddenLayer(word_lstm_dim, phono_char_lstm_dim,
                                                            name='final_phono_attention_input_layer', activation='tanh')
            # Evaluating attentional vector using a linear projection from final_output since the attention vector
            # must be conditioned on it and dimension must match the char lstm hidden dim.
            final_phono_attention_input = final_phono_attention_input_layer.link(final_output)
            phono_for_attention = self.get_TDAttention_vector(final_phono_attention_input,
                                                              phono_char_lstm_for.h.dimshuffle((1, 0, 2)),
                                                              phono_char_pos_ids)
            if char_bidirect:
                phono_rev_attention = self.get_TDAttention_vector(final_phono_attention_input,
                                                                  phono_char_lstm_rev.h.dimshuffle((1, 0, 2)),
                                                                  phono_char_pos_ids)
                attention_vectors.append(phono_rev_attention)
                attention_vector_size += phono_char_lstm_dim
            attention_vectors.append(phono_for_attention)
            attention_vector_size += phono_char_lstm_dim
        if len(attention_vectors) > 1:
            attention_vectors = T.concatenate(attention_vectors, axis=1)

        if use_phono_attention or use_ortho_attention:
            final_output = T.concatenate([final_output, attention_vectors], axis=1)
            post_word_lstm_output_size += attention_vector_size
            final_pre_crf_input_size += attention_vector_size



        # Sentence to Named Entity tags - Score
        final_layer = HiddenLayer(final_pre_crf_input_size, n_tags, name='final_layer',
                                  activation=(None if crf else 'softmax'))
        tags_scores = final_layer.link(final_output)

        # No CRF
        if not crf:
            cost = T.nnet.categorical_crossentropy(tags_scores, tag_ids).mean()
        # CRF
        else:
            transitions = shared((n_tags + 2, n_tags + 2), 'transitions')
            # n_tags + 2 to accommodate start and end symbols

            small = -1000 # = -log(inf)
            b_s = np.array([[small] * n_tags + [0, small]]).astype(np.float32)
            # Score of starting at start symbol is 1 => -log(1) = 0. Score of start symbol emitting any other NER
            # tag is -log(inf) = small
            e_s = np.array([[small] * n_tags + [small, 0]]).astype(np.float32)
            # Score of ending at end symbol is 1 => -log(1) = 0. Score of end symbol emitting any other NER
            # tag is -log(inf) = small
            observations = T.concatenate(
                [tags_scores, small * T.ones((s_len, 2))],
                axis=1
            )
            # observations is the emission energy (-log potential) between each token and each tag.
            # Emission score of intermediate words towards start and end tags is -log(inf)


            observations = T.concatenate(
                [b_s, observations, e_s],
                axis=0
            )
            # observations now contains the emission energies for start token, sentence tokens and end token


            # Score from tags
            real_path_score = tags_scores[T.arange(s_len), tag_ids].sum()
            # Sum of energies associated with the gold tags

            # Score from transitions
            b_id = theano.shared(value=np.array([n_tags], dtype=np.int32))
            e_id = theano.shared(value=np.array([n_tags + 1], dtype=np.int32))
            padded_tags_ids = T.concatenate([b_id, tag_ids, e_id], axis=0)
            real_path_score += transitions[
                padded_tags_ids[T.arange(s_len + 1)],
                padded_tags_ids[T.arange(s_len + 1) + 1]
            ].sum()
            # Transition scores from label_i to label_{i+1}

            all_paths_scores = forward(observations, transitions)
            cost = - (real_path_score - all_paths_scores)

        # Network parameters
        params = []
        if word_dim:
            self.add_component(word_layer)
            params.extend(word_layer.params)
        if ortho_char_dim:
            self.add_component(ortho_char_layer)
            self.add_component(ortho_char_lstm_for)
            params.extend(ortho_char_layer.params)
            params.extend(ortho_char_lstm_for.params)
            if char_bidirect:
                self.add_component(ortho_char_lstm_rev)
                params.extend(ortho_char_lstm_rev.params)

        if phono_char_dim:
            self.add_component(phono_char_layer)
            self.add_component(phono_char_lstm_for)
            params.extend(phono_char_layer.params)
            params.extend(phono_char_lstm_for.params)
            if char_bidirect:
                self.add_component(phono_char_lstm_rev)
                params.extend(phono_char_lstm_rev.params)

        if use_type_sparse_feats:
            self.add_component(type_level_sparse_layer)
            params.extend(type_level_sparse_layer.params)

        self.add_component(word_lstm_for)
        params.extend(word_lstm_for.params)

        if word_bidirect:
            self.add_component(word_lstm_rev)
            params.extend(word_lstm_rev.params)

        if word_bidirect or len(lstm_outputs) > 1:
            self.add_component(tanh_layer)
            params.extend(tanh_layer.params)

        if use_ortho_attention and ortho_char_dim:
            self.add_component(final_ortho_attention_input_layer)
            params.extend(final_ortho_attention_input_layer.params)
        if use_phono_attention and phono_char_dim:
            self.add_component(final_phono_attention_input_layer)
            params.extend(final_phono_attention_input_layer.params)

        self.add_component(final_layer)
        params.extend(final_layer.params)
        if crf:
            self.add_component(transitions)
            params.append(transitions)

        # Prepare train and eval inputs
        eval_inputs = []
        if word_dim:
            # eval_inputs.append(word_ids)
            eval_inputs.append(word_vecs)
        if ortho_char_dim:
            # eval_inputs.append(char_for_ids)
            eval_inputs.append(ortho_char_for_vecs)
            if char_bidirect:
                # eval_inputs.append(char_rev_ids)
                eval_inputs.append(ortho_char_rev_vecs)
            eval_inputs.append(ortho_char_pos_ids)
        if phono_char_dim:
            # eval_inputs.append(char_for_ids)
            eval_inputs.append(phono_char_for_vecs)
            if char_bidirect:
                # eval_inputs.append(char_rev_ids)
                eval_inputs.append(phono_char_rev_vecs)
            eval_inputs.append(phono_char_pos_ids)


        if use_type_sparse_feats:
            eval_inputs.append(type_sparse_feats)
        if use_token_sparse_feats:
            eval_inputs.append(token_sparse_feats)
        train_inputs = eval_inputs + [tag_ids]

        # Parse optimization method parameters
        if "-" in lr_method:
            lr_method_name = lr_method[:lr_method.find('-')]
            lr_method_parameters = {}
            for x in lr_method[lr_method.find('-') + 1:].split('-'):
                split = x.split('_')
                assert len(split) == 2
                lr_method_parameters[split[0]] = float(split[1])
        else:
            lr_method_name = lr_method
            lr_method_parameters = {}

        # Compile training function
        print 'Compiling...'
        if training:
            updates = Optimization(clip=5.0).get_updates(lr_method_name, cost, params, **lr_method_parameters)
            f_train = theano.function(
                inputs=train_inputs,
                outputs=cost,
                updates=updates,
                givens=({is_train: np.cast['int32'](1)} if dropout else {})
            )
        else:
            f_train = None

        # Compile evaluation function
        if not crf:
            f_eval = theano.function(
                inputs=eval_inputs,
                outputs=tags_scores,
                givens=({is_train: np.cast['int32'](0)} if dropout else {})
            )
        else:
            f_eval = theano.function(
                inputs=eval_inputs,
                outputs=forward(observations, transitions, viterbi=True,
                                return_alpha=False, return_best_sequence=True),
                givens=({is_train: np.cast['int32'](0)} if dropout else {})
            )
        print("Finished Compiling")
        return f_train, f_eval
