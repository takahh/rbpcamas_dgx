# -------------------------------------------------------------------
# Created on 2021.11.4
# Statistical potential table to augument attention
# was written, so now train with attention augumented
# with the potentials
# -------------------------------------------------------------------

# ------------------------------------------------
# Import
# ------------------------------------------------
import time

import numpy as np
import tensorflow as tf
import os
import datetime
import tensorflow_addons as tfa

#########################
# Horovod, GPU Setup
#########################

seed = 1
tf.config.run_functions_eagerly(True)
tf.random.set_seed(seed)
operation_lebel_seed = 1
initializer = tf.keras.initializers.GlorotUniform(seed=operation_lebel_seed)

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------

########################################################################
# transformer
# variables to consider
########################################################################

PCODES = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
RCODES = ["A", "G", "C", "U"]
WARMUP_STEPS = 5
USE_CHECKPOINT = False
TRAINING = True
MAX_EPOCHS = 10
D_MODEL = 64
DFF = 64
RNA_MAX_LENGTH = 101
# PROTEIN_MAX_LENGTH = 2805
# DROPOUT_RATE = 0.1
final_target_vocab_size = 2
ATTN_OUT = "/home/kimura.t/rbpcamas/attn_out/"

seed = 0
tf.config.run_functions_eagerly(True)
tf.random.set_seed(seed)
operation_lebel_seed = 0
initializer = tf.keras.initializers.GlorotUniform(seed=operation_lebel_seed)

class Tfconfig():
    def __init__(self):
        # self.max_pro_len = PROTEIN_MAX_LENGTH
        self.max_rna_len = RNA_MAX_LENGTH
        self.num_layers = None
        self.d_model = D_MODEL
        self.num_heads = None
        self.dff = DFF
        self.dropout_rate = None
        self.group_to_ignore = None
        self.max_epoch = MAX_EPOCHS
        self.datapath = None
        self.init_lr = None
        self.benchname = None
        self.pair_name = None
        self.basepath = None
        self.pvecs_list_padded = None
        self.rna_vec_list = None
        self.label_list = None
        self.pro_padding_mask_list = None
        self.rna_padding_mask_list = None
        self.cross_padding_mask_list = None
        self.cross_padding_mask_list_small = None
        self.enc_output = None
        self.self_pro_mask_list = None
        self.self_rna_mask_list = None
        self.rnaseq = None
        self.proseq = None
        self.accum_gradient = None
        self.proid = None
        self.protok = None
        self.rnatok = None
        self.label = None
        self.statpot_pi = None
        self.statpot_hb = None
        self.run_on_local = None
        self.warmup_steps = 0
        self.training_boolean = None
        self.aug_or_not = None
        self.record_count = 0
        self.step = 0
        self.num_of_node = 1
        self.num_of_gpu = 1
        self.aug_weight_aug = None
        self.aug_weight_at = None
        self.predictions = None
        self.labels = None
        self.losssum = 0
        self.batch_count = 0
        self.aug_multiply = 0
        self.only_rna_path = 0
        self.only_protein_path = 0
        self.red_index = None
        self.test_batch_size = None
        self.reduced_pro_tokenlist = None
        self.validation_in_training = 0
        self.test_freq = 1

    def update(self, task_id):
        # -------------
        # gpu number
        # -------------
        self.test_freq = int(self.batch_size / 2)
        if self.node_name == "f":
            self.num_of_gpu = self.num_of_node * 4
        elif self.node_name == "q":
            self.num_of_gpu = self.num_of_node
        else:
            print("Node name is incorrect !!!")
        # -------------
        # local or remote
        # -------------
        if self.run_on_local == 1:
            self.basepath = "/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/"
            physical_devices = tf.config.list_physical_devices('CPU')  # 8 GPUs in my setup
            tf.config.set_visible_devices(physical_devices[0], 'CPU')  # Using all GPUs (default behaviour)
        elif self.run_on_local == 0:
            self.basepath =  "/home/kimura.t/rbpcamas/"
            gpus = tf.config.list_physical_devices('GPU')  # 8 GPUs in my setup
            tf.config.set_visible_devices(gpus[0:config.num_of_gpu], 'GPU')  # Using all GPUs (default behaviour)
        # -------------
        # train or test
        # -------------
        if self.training == 0:
            self.max_epoch = 1
            self.training_boolean = False
        else:
            self.training_boolean = True
        # -------------
        # input path
        # -------------
        self.training_npz_dir = f"{self.basepath}data/{self.data_dir_name}"

        # -------------
        # chpoint path
        # -------------
        if self.training == 1:
            self.checkpoint_path = f"{self.basepath}chpoint/{self.keyword}/chpt_"
        elif self.training == 0:
            self.checkpoint_path = f"{self.basepath}chpoint/{self.keyword}/chpt_"
        self.checkpoint_dir = f"{self.basepath}chpoint/{self.keyword}/"
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        # self.data_for_auc = self.basepath + "/data/for_comparison/CrossRPBmer/"
        promaxdict = {20: 2805, 13: 1609, 8: 986, 4: 609}
        self.protein_reduced_max_length = promaxdict[self.reduce_level]
        return self

    def add_rawdata(self, pred, label):
        if self.predictions is None:
            self.predictions = pred
        else:
            self.predictions = np.concatenate([self.predictions, pred])
        if self.labels is None:
            self.labels = label
        else:
            self.labels = np.concatenate([self.labels, label])
        return self


# ------------------------------------------------
# Functions
# ------------------------------------------------


@tf.function
def scaled_dot_product_attention(self, k, q, v, tffig, one_on_rna):
    ########################################################
    # Calculate Q x K
    ########################################################

    # one_on_rna = 1:rna-self, 0:protein-self, 2:cross
    matmul_qk = tf.cast(tf.matmul(q, k, transpose_b=True), tf.float32)  # (..., seq_len_q, seq_len_k)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    # assign smask in Decoder's 1st layer
    if one_on_rna == 1:  # rna self
        mask = None
    elif one_on_rna == 0:  # pro self
        mask = tffig.self_pro_mask_list
    else:  # cross
        mask = tffig.cross_padding_mask_list
    v = tf.cast(v, tf.float32)
    scaled_attention_logits = matmul_qk
    multiply_num = tf.cast(tffig.num_heads / 2, dtype="int32")

    ########################################################
    # divide with squared root of dimension of leading vectors
    ########################################################
    scaled_attention_logits /= tf.math.sqrt(dk)
    ########################################################
    # add or multiply statistical potentials at Cross Attention Layers
    ########################################################
    if one_on_rna == 2:
        if tffig.use_attn_augument == 1:  # in a cross layer and the aug flag is on
            # ------------------------------------------
            # Prepare Table to Add
            # ------------------------------------------
            table_to_augment_hb = tffig.statpot_hb
            table_to_augment_pi = tffig.statpot_pi  # TensorShape([5, 2805, 101])
            table_to_augment = tf.transpose(tf.stack([table_to_augment_hb, table_to_augment_pi]),
                                            perm=[1, 0, 2, 3])  # [5, 2, 2805, 101]
            try:
                table_to_augment = tf.repeat(table_to_augment, repeats=[multiply_num, multiply_num],
                                             axis=1)  # [5, 4, 2805, 101]  THIS LINE IS WRONG!!!!
            except ValueError:
                table_to_augment = tf.repeat(table_to_augment, repeats=[multiply_num, multiply_num + 1], axis=1)
            if scaled_attention_logits.shape[-1] != table_to_augment.shape[-1]:
                table_to_augment = tf.transpose(table_to_augment, perm=[0, 1, 3, 2])

            # ------------------------------------------
            # Augment (Add or Multiply)
            # ------------------------------------------
            scaled_attention_logits *= self.aug_weight_att
            attn_weight_before = scaled_attention_logits

            if tffig.aug_multiply == 0:  # when add two tables
                # # ------------------------------------------
                # # Scale KQ
                # # ------------------------------------------
                # scaled_attention_logits /= 100
                # ------------------------------------------
                # Apply coeff to potentials
                # ------------------------------------------
                stats_to_add = self.aug_weight_aug * 100 * tf.cast(table_to_augment, dtype="float32")
                # tf.print("#######")
                # tf.print(f"stats_to_add {stats_to_add.shape}")
                # tf.print(f"stats_to_add {stats_to_add.numpy()[0, :3, :10]}")
                # tf.print(f"scaled_attention_logits {scaled_attention_logits.shape}")
                # tf.print(f"scaled_attention_logits {scaled_attention_logits.numpy()[0, :3, :10]}")
                # ------------------------------------------
                # Add
                # ------------------------------------------
                scaled_attention_logits += stats_to_add
            else:  # multiply two tables element-wise
                table_to_augment = 10 * tf.cast(table_to_augment, dtype="float32")
                table_to_augment = tf.keras.activations.sigmoid(table_to_augment)
                scaled_attention_logits = tf.keras.activations.sigmoid(scaled_attention_logits)
                scaled_attention_logits = \
                    tf.math.multiply(tf.cast(table_to_augment, dtype="float32"), scaled_attention_logits)

    ########################################################
    # add large negative values 7
    ########################################################
    if one_on_rna != 1:  # except RNA
        if one_on_rna == 2:  # when cross, do transpose if needed
            if scaled_attention_logits.shape[-1] != mask.shape[-1]:
                mask = tf.transpose(mask, perm=[0, 2, 1])
        mask = tf.repeat(mask[:, tf.newaxis, :, :], repeats=tffig.num_heads, axis=1)
        scaled_attention_logits += (mask * -1e9)
        # tf.print(f"scaled_attention_logits after added large nega {scaled_attention_logits.shape}--{scaled_attention_logits.numpy()[0, :3, :10]}")

    ########################################################
    # apply softmax
    ########################################################
    # if tffig.use_attn_augument == 1:
    if tffig.two_d_softm == 1:
        attention_weights = tf.cast(tf.keras.activations.softmax(scaled_attention_logits, axis=[2, 3]), tf.float32)
        # multiply the number of rows (necessary??)
        if tffig.two_d_softm_mul_row_count == 1:
            if attention_weights.shape[2] == tffig.max_pro_len:  # (5, 105, 2805)
                mul_coeff = tf.math.count_nonzero(tffig.reduced_ptok, 2)[0][:, tf.newaxis, tf.newaxis, tf.newaxis]
                # mulcoeff (5, 1, 1, 1). att weights [5,2,1668,1668]
                attention_weights *= tf.cast(mul_coeff, dtype="float32")
            elif attention_weights.shape[2] == 101:
                attention_weights *= 101
    else:
        # tf.print(f"scaled_attention_logits {scaled_attention_logits.shape}--{scaled_attention_logits.numpy()[0, :3, :10]}")
        attention_weights = tf.cast(tf.keras.activations.softmax(scaled_attention_logits, axis=3), tf.float32)
    # else:
    #     attention_weights = tf.cast(tf.keras.activations.softmax(scaled_attention_logits, axis=3), tf.float32)
    ########################################################
    # calculate weight matrix x V
    ########################################################
    output = tf.matmul(tf.transpose(attention_weights, perm=[0, 1, 3, 2]), v)  # (..., seq_len_q, depth_v)

    ########################################################
    # pack matrices of cross attention
    ########################################################
    if one_on_rna == 2 and tffig.use_attn_augument == 1 and tffig.aug_multiply == 0:
        attention_weights = [attn_weight_before, stats_to_add, attention_weights]

    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(
            dff, activation='relu',
            kernel_initializer=initializer),
        # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(
            d_model,
            kernel_initializer=initializer)
        # (batch_size, seq_len, d_model)
    ])


class FinalLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(FinalLayer, self).__init__()
        self.finallayer = tf.keras.layers.Dense(
            final_target_vocab_size,
            activation='softmax',
            kernel_initializer=initializer)

    def call(self, x1, x2, cffg):  # x1 : rna, x2 : pro
        if cffg.only_protein_path == 1:
            x = tf.reduce_mean(x2, axis=1)
        elif cffg.only_rna_path == 1:
            x = tf.reduce_mean(x1, axis=1)
        else:
            x = tf.keras.layers.Average()([tf.reduce_mean(x1, axis=1), tf.reduce_mean(x2[0], axis=1)])
        x = self.finallayer(x)
        return x


class Transformer(tf.keras.Model):
    def __init__(self, trcfg):
        super(Transformer, self).__init__()
        self.p_embedders = Embedders(trcfg, 0)
        self.r_embedders = Embedders(trcfg, 1)
        self.p_encoders = Encoders(trcfg, 0)
        self.r_encoders = Encoders(trcfg, 1)
        self.cross_layers = CrossLayers(trcfg)
        self.final_layer = FinalLayer()

    def call(self, tf_cfg):
        pout = self.p_embedders.call(tf_cfg, 0)
        rout = self.r_embedders.call(tf_cfg, 1)
        pout, pweight = self.p_encoders.call(tf_cfg, pout)
        rout, rweight = self.r_encoders.call(tf_cfg, rout)
        pout, rout, p_cr_weight, r_cr_weight = self.cross_layers.call(tf_cfg, rout, pout)
        prediction = self.final_layer.call(rout[0], pout, tf_cfg)
        if tf_cfg.training == 1:
            return prediction
        else:
            return prediction, p_cr_weight, r_cr_weight, pweight, rweight, pout, rout


def padding_to_zeros(input_tf, tffcfg):
    boolmask = tf.logical_not(tf.math.equal(tffcfg.reduced_ptok, tf.constant(0.0)))
    boolmask = boolmask[:, :, :, tf.newaxis]
    boolmask = tf.repeat(boolmask, repeats=64, axis=3)
    zero_tf = tf.zeros(input_tf.shape)
    output_tf = tf.where(boolmask, input_tf, zero_tf)
    return output_tf


class Embedders(tf.keras.layers.Layer):
    def __init__(self, encfg, one_if_rna):
        super(Embedders, self).__init__()
        if one_if_rna == 1:  # RNA self
            vocab_size = 4
            maxlen = encfg.max_rna_len
            self.embedding = tf.keras.layers.Embedding(vocab_size + 1, encfg.d_model,
                                                       mask_zero=True)  # 5 because 4 bases and an unknown
        elif one_if_rna == 0:  # protein self (no cross in embed)
            maxlen = 2805
            vocab_size = encfg.reduce_level
            if encfg.use_TAPE_feature == 1:
                self.dense = tf.keras.layers.Dense(
                    encfg.d_model,
                    # activation='tanh',
                    kernel_initializer=initializer)
            else:
                self.embedding = tf.keras.layers.Embedding(vocab_size + 1, encfg.d_model,
                                                           mask_zero=True)  # 5 because 4 bases and an unknown
        self.d_model = tf.cast(encfg.d_model, tf.float32)
        self.pos_encoding = positional_encoding(maxlen, encfg.d_model)
        self.dropout = tf.keras.layers.Dropout(encfg.dropout_rate)

    def call(self, tf_cfg, one_when_rna):  # tfcfg, RCODES, i, 1, self.rna_out
        if one_when_rna == 1:  # RNA embedding
            sequence = tf_cfg.rnatok
            maxlen = tf_cfg.max_rna_len
            x = self.embedding(sequence)
        elif one_when_rna == 0:  # protein embedding
            sequence = tf_cfg.reduced_ptok
            maxlen = 2805
            # ----------------------------------
            # EMBEDDING
            # ----------------------------------
            if tf_cfg.use_TAPE_feature == 1:
                x = self.dense(tf_cfg.p_tape_tf)
            # embedding
            else:
                x = self.embedding(sequence)   # move this after posi_enc  EMBEDDING
                x = x[0]
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # x=(1, 662, 64)
        # ----------------------------------
        # ADD POSITIONAL ENCODiNG
        # ----------------------------------
        if one_when_rna == 0:  # protein   (5, 2805, 64)
            # add posi_enc to x after slicing the posi_enc
            # self.pos_encoding (1, 2805, 64)
            # tf_cfg.red_index  (1, 5, 2805)
            posi_info = tf.gather(self.pos_encoding[0, :, :], tf_cfg.red_index, axis=0)[:, :, :tf_cfg.max_pro_len, :]
            x += posi_info
            # padding to zeros
            x = padding_to_zeros(x, tf_cfg)
        else:  # RNA
            x += self.pos_encoding[:maxlen, :]

        # dropout
        x = self.dropout(x, training=tf_cfg.training_boolean)
        x = tf.cast(x, tf.float32)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x /= tf.cast(self.d_model, tf.float32)

        return x


class Encoders(tf.keras.layers.Layer):
    def __init__(self, trscfg, one_if_rna):
        super(Encoders, self).__init__()
        if one_if_rna == 1:
            self.layer_num = trscfg.self_rlayer_num
        else:
            self.layer_num = trscfg.self_player_num
        self.rencoder = [SelfAttention(trscfg, one_if_rna) for _ in range(self.layer_num)]

    def call(self, tfcfg, x):
        weights = {}
        for i in range(self.layer_num):
            x, weights[f"layer_{i + 1}"] = self.rencoder[i].call(tfcfg, x)
        return x, weights


class CrossLayers(tf.keras.layers.Layer):
    def __init__(self, trscfg):
        super(CrossLayers, self).__init__()
        for i in range(trscfg.cross_layer_num):
            self.cross_attention = CrossAttention(trscfg)

    def call(self, tfcfg, rna_out, pro_out):
        pro_cross_weights, rna_cross_weights = {}, {}
        for i in range(tfcfg.cross_layer_num):
            pro_out, rna_out, pro_cross_weights[f"layer_{i + 1}"], rna_cross_weights[f"layer_{i + 1}"] \
                = self.cross_attention.call(tfcfg, pro_out, rna_out)  # q, kv, confg, layr_num, one_when_rna
        return pro_out, rna_out, pro_cross_weights, rna_cross_weights


class CrossAttention(tf.keras.layers.Layer):  # two inputs/outputs
    def __init__(self, cr_cfg):
        super(CrossAttention, self).__init__()
        if cr_cfg.only_rna_path == 1:
            self.rna_cross_attention = AttentionLayer(cr_cfg, cr_cfg.cross_dff)
        else:
            if cr_cfg.only_protein_path == 1:
                self.pro_cross_attention = AttentionLayer(cr_cfg, cr_cfg.cross_dff)
            else:
                self.pro_cross_attention = AttentionLayer(cr_cfg, cr_cfg.cross_dff)
                self.rna_cross_attention = AttentionLayer(cr_cfg, cr_cfg.cross_dff)
    def call(self, cross_cfg, pro_inp, rna_inp):
        # q, kv, confg, layr_num, one_when_rna
        if cross_cfg.only_rna_path == 1:
            rna_out, rna_cross_weights = self.rna_cross_attention.call(rna_inp, pro_inp, cross_cfg, 2)  # q, kv
            pro_out = pro_inp
            pro_cross_weights = {}
        else:
            if cross_cfg.only_protein_path == 1:
                rna_out = rna_inp
                rna_cross_weights = {}
                pro_out, pro_cross_weights = self.pro_cross_attention.call(pro_inp, rna_inp, cross_cfg, 2)
            else:
                pro_out, pro_cross_weights = self.pro_cross_attention.call(pro_inp, rna_inp, cross_cfg, 2)
                rna_out, rna_cross_weights = self.rna_cross_attention.call(rna_inp, pro_inp, cross_cfg, 2)  # q, kv
        return pro_out, rna_out, pro_cross_weights, rna_cross_weights


class SelfAttention(tf.keras.layers.Layer):  # one input/output
    def __init__(self, encfg, one_if_rna):
        super(SelfAttention, self).__init__()
        self.one_if_rna = one_if_rna
        if one_if_rna == 1:
            dff = encfg.rna_dff
        else:
            dff = encfg.pro_dff
        self.self_layer = AttentionLayer(encfg, dff)
        # self.dropout = tf.keras.layers.Dropout(encfg.dropout_rate)

    def call(self, tf_cfg, x=None):  # tfcfg, RCODES, i, 1, self.rna_out
        x, self_weights = self.self_layer.call(x, x, tf_cfg, self.one_if_rna)
        return x, self_weights


class AttentionLayer(tf.keras.layers.Layer):  # basic attention calculation
    def __init__(self, elcfg, dff):
        super(AttentionLayer, self).__init__()
        self.mha = MultiHeadAttention(elcfg)
        self.ffn = point_wise_feed_forward_network(elcfg.d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-9)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-9)
        self.dropout1 = tf.keras.layers.Dropout(elcfg.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(elcfg.dropout_rate)

    def call(self, q, kv, confg, one_when_rna):
        attn_output, _ = self.mha.call(q, kv, confg, one_when_rna)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=confg.training_boolean)
        out1 = self.layernorm1(q + attn_output)  # (batch_size, input_seq_len, d_model)
        if out1.shape[-2] == config.max_pro_len:
            out1 = padding_to_zeros(out1, confg)
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        if ffn_output.shape[2] == confg.max_pro_len:
            ffn_output = padding_to_zeros(ffn_output, confg)
        ffn_output = self.dropout2(ffn_output, training=confg.training_boolean)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        if out2.shape[2] == confg.max_pro_len:
            out2 = padding_to_zeros(out2, confg)
        return out2, _


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, mhacfg):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = mhacfg.num_heads
        self.d_model = mhacfg.d_model
        assert mhacfg.d_model % self.num_heads == 0
        self.depth = mhacfg.d_model // self.num_heads  # depth = 128 /4 = 32
        self.wq = tf.keras.layers.Dense(mhacfg.d_model, kernel_initializer=initializer)
        self.wk = tf.keras.layers.Dense(mhacfg.d_model, kernel_initializer=initializer)
        self.wv = tf.keras.layers.Dense(mhacfg.d_model, kernel_initializer=initializer)
        self.dense = tf.keras.layers.Dense(mhacfg.d_model, kernel_initializer=initializer)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-9)
        with strategy.scope():
            # define coeffs for augmentation
            if mhacfg.use_attn_augument == 1:
                if mhacfg.clip_coeff == 1:
                    self.aug_weight_att = tf.compat.v1.get_variable(
                        "aug_weight_att",
                        shape=[1],
                        trainable=True,
                        dtype=tf.float32,
                        constraint=lambda x: tf.clip_by_value(x, 0, 1),
                        initializer=tf.constant_initializer(1))
                    self.aug_weight_aug = tf.compat.v1.get_variable(
                        "aug_weight_aug",
                        shape=[1],
                        trainable=True,
                        dtype=tf.float32,
                        constraint=lambda x: tf.clip_by_value(x, 0, 1),
                        initializer=tf.constant_initializer(1))
                else:
                    self.aug_weight_att = tf.compat.v1.get_variable(
                        "aug_weight_att",
                        shape=[1],
                        trainable=True,
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(1))
                    self.aug_weight_aug = tf.compat.v1.get_variable(
                        "aug_weight_aug",
                        shape=[1],
                        trainable=True,
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(mhacfg.initial_aug_coeff))
                tf.compat.v1.get_variable_scope().reuse_variables()
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, kv, tfcg, one_for_rna):
        if tfcg.validation_in_training == 0:
            batch_size = int(tfcg.batch_size * 5 / 4)
        elif tfcg.validation_in_training == 1:
            batch_size = int(tfcg.test_batch_size * 5 / 4)
        v = self.wv(kv)
        k = self.wk(kv)
        q = self.wq(q)
        if v.shape[2] == tfcg.max_pro_len:
            v = padding_to_zeros(v, tfcg)
            k = padding_to_zeros(k, tfcg)
        if q.shape[2] == tfcg.max_pro_len:
            q = padding_to_zeros(q, tfcg)
        # split
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        # calc. attention

        if tfcg.use_attn_augument == 1:
            scaled_attention, attention_weights = scaled_dot_product_attention(self, q, k, v, tfcg, one_for_rna)
            if one_for_rna == 2 and tfcg.validation_in_training == 1:
                tf.print(f"coeff att {self.aug_weight_att.numpy()}, coeff aug {self.aug_weight_aug.numpy()}")
        else:
            scaled_attention, attention_weights = scaled_dot_product_attention(self, q, k, v, tfcg, one_for_rna)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        if output.shape[1] == tfcg.max_pro_len:
            output = padding_to_zeros(output, tfcg)
        return output, attention_weights


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        self.lr = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        return self.lr


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def opt_trfmr(tfconfig):
    # ------------------------------------------------
    # functions and else
    # ------------------------------------------------
    def loss_function(labellist, predictions):
        bce = tf.keras.metrics.binary_crossentropy(labellist, predictions)
        lossvalue = tf.cast(bce, dtype="float32")
        return lossvalue

    learning_rate = tfconfig.init_lr
    class CustomModel(tf.keras.Model):

        def __init__(self, tffconfig):
            super(CustomModel, self).__init__()
            self.transformer = Transformer(tffconfig)
            # define coefficients for attention augmentation
            #     tfconfig.aug_weight_aug = self.aug_weight_aug
            #     tfconfig.aug_weight_at = self.aug_weight_at

        def flatten_n_batch(self, x):
            dshape = x.shape
            if len(dshape) == 3:
                return tf.reshape(x, [1, dshape[0] * dshape[1], dshape[2]])
            else:
                return tf.reshape(x, [dshape[0] * dshape[1], dshape[2], dshape[3]])

        @tf.function()
        # @tf.function(input_signature=train_step_signature)
        def train_step(self, data_combined, y=None, first_batch=None):
            datalist = [x for x in data_combined]

            tfconfig.proid, tfconfig.rnatok, tfconfig.statpot_hb, tfconfig.statpot_pi, \
            tfconfig.self_pro_mask_list, tfconfig.cross_padding_mask_list, tfconfig.label, tfconfig.red_index \
            , tfconfig.reduced_ptok = \
            map(self.flatten_n_batch, datalist[:9])
            tfconfig.validation_in_training = 0

            # tfconfig.proid = self.flatten_n_batch(datalist[0])
            # # tfconfig.protok = self.flatten_n_batch(datalist[1]) # 1 is skipped
            # tfconfig.rnatok = self.flatten_n_batch(datalist[1])
            # tfconfig.statpot_hb = self.flatten_n_batch(datalist[2])
            # tfconfig.statpot_pi = self.flatten_n_batch(datalist[3])
            # tfconfig.self_pro_mask_list = self.flatten_n_batch(datalist[4])
            # tfconfig.cross_padding_mask_list = self.flatten_n_batch(datalist[5])
            # tfconfig.label = self.flatten_n_batch(datalist[6])
            # tfconfig.red_index = self.flatten_n_batch(datalist[7])
            # tfconfig.reduced_ptok = self.flatten_n_batch(datalist[8])

            # namelist = ["proid", "protok", "rnatok", "ststhb", "statpi", "promask", "crossmask", "label"]
            # varlist = [tfconfig.proid, tfconfig.protok, tfconfig.rnatok, tfconfig.statpot_hb, tfconfig.statpot_pi, tfconfig.self_pro_mask_list, tfconfig.cross_padding_mask_list, tfconfig.label]
            # for idx, item in enumerate(varlist) :
            #     tf.print(f"{namelist[idx]}: {item[0, 0]}")

            if tfconfig.use_TAPE_feature == 1:
                tfconfig.p_tape_tf = self.flatten_n_batch(datalist[9])
            with tf.GradientTape() as tape:
                predictions = self.transformer(tfconfig)  # prediction, pweight, rweight, p_cr_weight, r_cr_weight
                loss = loss_function(tfconfig.label[0], predictions)
            if tfconfig.roc_data_log == 1:
                tf.print(f"train ROC_DATA:" + ":" + str(predictions.numpy()).replace('\n', '') + "-" + str(
                    tfconfig.label).replace('\n', ''))
            loss_avg.update_state(loss)
            gradients = tape.gradient(loss, self.trainable_variables)
            auc.update_state(tfconfig.label[0], predictions)
            tfconfig.losssum += np.sum(loss.numpy())
            if tfconfig.training == 1:
                trainable_variables = [var for grad, var in zip(gradients, self.transformer.trainable_variables) if
                                       grad is not None]
                gradient = [flat_gradients(grad) for grad in gradients if grad is not None]
                optimizer.apply_gradients(zip(gradient, trainable_variables))
                # for name, value in zip(trainable_variables, gradient):
                #     tf.print(f"{name}: {value}")
            tfconfig.batch_count += 1
            # return loss
            tf.print(f"loss_avg.result() {loss_avg.result()}")
            tf.print(f"auc.result() {auc.result()}")
            return {"loss": loss_avg.result(), "auc": auc.result()}

        @tf.function()
        # @tf.function(input_signature=train_step_signature)
        def test_step(self, data_combined, y=None, first_batch=None):
            datalist = [x for x in data_combined]

            tfconfig.proid, tfconfig.rnatok, tfconfig.statpot_hb, tfconfig.statpot_pi, \
            tfconfig.self_pro_mask_list, tfconfig.cross_padding_mask_list, tfconfig.label, tfconfig.red_index \
                    , tfconfig.reduced_ptok = \
                        map(self.flatten_n_batch, datalist[:9])
            tfconfig.validation_in_training = 0
            tfconfig.protok = None

            # tfconfig.proid = self.flatten_n_batch(datalist[0])
            # # tfconfig.protok = self.flatten_n_batch(datalist[1])
            # tfconfig.protok = None
            # tfconfig.rnatok = self.flatten_n_batch(datalist[1])
            # tfconfig.statpot_hb = self.flatten_n_batch(datalist[2])
            # tfconfig.statpot_pi = self.flatten_n_batch(datalist[3])
            # tfconfig.self_pro_mask_list = self.flatten_n_batch(datalist[4])
            # tfconfig.cross_padding_mask_list = self.flatten_n_batch(datalist[5])
            # tfconfig.label = self.flatten_n_batch(datalist[6])
            # tfconfig.red_index = self.flatten_n_batch(datalist[7])
            # tfconfig.reduced_ptok = self.flatten_n_batch(datalist[8])
            tfconfig.validation_in_training = 1
            if tfconfig.use_TAPE_feature == 1:
                tfconfig.p_tape_tf = self.flatten_n_batch(datalist[9])
            # predictions, pweight, rweight, pself, rself, proout, rnaout = self.transformer(tfconfig)
            predictions = self.transformer(tfconfig)

            # predictions, pweight, rweight, pself, rself = self.transformer(tfconfig)  # prediction, pweight, rweight, p_cr_weight, r_cr_weight
            loss = loss_function(tfconfig.label[0], predictions)
            loss_avg.update_state(loss)
            auc.update_state(tfconfig.label[0], predictions)
            tfconfig.batch_count += 1
            if not os.path.exists(f"{ATTN_OUT}/{tfconfig.data_dir_name}"):
                os.mkdir(f"{ATTN_OUT}/{tfconfig.data_dir_name}")
            if tfconfig.training == 0:
                np.save(f"{ATTN_OUT}/{tfconfig.data_dir_name}/attn_analysis_hb_proout", proout)
                np.save(f"{ATTN_OUT}/{tfconfig.data_dir_name}/attn_analysis_hb_RNAout", rnaout)

                tf.print(np.save(f"{ATTN_OUT}/{tfconfig.data_dir_name}/attn_analysis_hb_pro", np.array(pweight["layer_1"])))
                tf.print(np.save(f"{ATTN_OUT}/{tfconfig.data_dir_name}/attn_analysis_hb_rna", np.array(rweight["layer_1"])))

                for i in range(4):
                    np.save(f"{ATTN_OUT}/{tfconfig.data_dir_name}/attn_analysis_hb_pro_self_l_{i + 1}",
                            np.array(pself[f"layer_{i + 1}"]))
                    np.save(f"{ATTN_OUT}/{tfconfig.data_dir_name}/attn_analysis_hb_rna_self_l_{i + 1}",
                            np.array(rself[f"layer_{i + 1}"]))
            if tfconfig.roc_data_log == 1:
                tf.print("test ROC_DATA:" + ":" + str(predictions.numpy()).replace('\n', '') + "-" + str(
                    tfconfig.label).replace('\n', ''))
            return {"test_auc": auc.result(), "test_loss": loss_avg.result()}

        @property
        def metrics(self):
            # loss_avg = tf.keras.metrics.Mean(name='train_loss')
            # auc = tf.keras.metrics.AUC()
            return [loss_avg, auc]

    def get_unknown_npzlist(fold_id):  # return tf.dataset consisting of 800 training and 200 test
        # /transformer_tape_dnabert/data_lncRNA/training_data_tokenized/0
        four_dataset = None
        five_dataset = None
        first_done = 0
        # add training set four times
        for i in range(5):
            if i != fold_id:
                groupdirpath = f"{tfconfig.training_npz_dir}/{i}/"
                if first_done == 0:
                    four_dataset = tf.data.Dataset.list_files(groupdirpath + "*.npz")
                    first_done = 1
                else:
                    four_dataset = four_dataset.concatenate(tf.data.Dataset.list_files(groupdirpath + "*.npz"))
        # add test set at the end
        five_dataset = four_dataset.concatenate(
            tf.data.Dataset.list_files(f"{tfconfig.training_npz_dir}{fold_id}/" + "*.npz"))
        return five_dataset

    def np_load(filename):
        # proid, protok, rnatok, p_tape_arr, label, pro_mask, cross_mask, hb_pots, pi_pots, reduced_index)
        arr = np.load(filename.numpy(), allow_pickle=True)
        proid_tf = tf.convert_to_tensor(arr["proid"], dtype="int16")
        protok_tf = None
        rnatok_tf = tf.convert_to_tensor(arr["rnatok"], dtype="float32")
        pro_mask_tf = tf.convert_to_tensor(arr["pro_mask"], dtype="float32")
        cross_mask_tf = tf.convert_to_tensor(arr["cross_mask"], dtype="float32")
        pot_arr_hb_tf = tf.convert_to_tensor(arr["hb_pots"], dtype="float32")
        pot_arr_pi_tf = tf.convert_to_tensor(arr["pi_pots"], dtype="float32")
        label_tf = tf.convert_to_tensor(arr["label"], dtype="int16")
        red_index_tf = tf.convert_to_tensor(arr["reduced_index"], dtype="int32")
        reduced_ptoks_tf = tf.convert_to_tensor(arr["reduced_ptoks"], dtype="float32")
        if tfconfig.use_TAPE_feature == 1:
            p_tape_tf = tf.convert_to_tensor(arr["p_tape_arr"], dtype="float32")
            return (proid_tf, rnatok_tf, pot_arr_hb_tf, pot_arr_pi_tf, pro_mask_tf, cross_mask_tf, label_tf, red_index_tf,
                    reduced_ptoks_tf, p_tape_tf)
        else:
            return (proid_tf, rnatok_tf, pot_arr_hb_tf, pot_arr_pi_tf, pro_mask_tf, cross_mask_tf, label_tf, red_index_tf,
                    reduced_ptoks_tf)

    # ------------------------------------------------
    # callbacks and dataset
    # ------------------------------------------------

    starttime = time.process_time()
    # optimizer = tf.keras.optimizers.Adam(learning_rate)
    optimizer = tfa.optimizers.RectifiedAdam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9,
                                             weight_decay=1e-5)
    current_time = datetime.datetime.now().strftime("%Y%m%d")
    checkpoint_path_no_date = f"{tfconfig.checkpoint_path}"
    checkpoint_path_with_date = f"{tfconfig.checkpoint_path}_{current_time}"
    if not os.path.exists(tfconfig.checkpoint_dir):
        try:
            os.mkdir(tfconfig.checkpoint_dir)
        except FileExistsError:
            pass
    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, log=None, log2=None):
            auc.reset_states()
            loss_avg.reset_states()
            endtime = time.process_time()

    cp_callback_no_date = tf.keras.callbacks.ModelCheckpoint(checkpoint_path_no_date,
                                                     monitor="eval_loss",
                                                     mode="min",
                                                     save_weights_only=True,
                                                     verbose=1)
    cp_callback_with_date = tf.keras.callbacks.ModelCheckpoint(checkpoint_path_with_date,
                                                     monitor="eval_loss",
                                                     mode="min",
                                                     save_weights_only=True,
                                                     verbose=1)

    # get paths
    npz_list = None
    if tfconfig.data_mode == "mydata":
        npz_list = tf.data.Dataset.list_files(tfconfig.training_npz_dir + "*.npz")
        train_files = 800
        test_files = 200
        val_files = 0
    elif tfconfig.data_mode == "unknown":
        npz_list = get_unknown_npzlist(tfconfig.cv_fold_id)
        train_files = 2640
        test_files = 660
        val_files = 0
    elif tfconfig.data_mode == "increased_red_6":
        npz_list = tf.data.Dataset.list_files(tfconfig.training_npz_dir + "*.npz")
        train_files = 3000
        test_files = 500
        val_files = 10
    elif tfconfig.data_mode == "unknown_data_to_known_task":
        npz_list = tf.data.Dataset.list_files(tfconfig.training_npz_dir + "*.npz")
        train_files = 800
        test_files = 200
        val_files = 0
    elif tfconfig.data_mode == "new_old_mix":
        tf.print("datamode in new_old_mix ####################")
        npz_list_pro = tf.data.Dataset.list_files(tfconfig.training_npz_dir.split(":")[0] + "*.npz")
        npz_list_rna = tf.data.Dataset.list_files(tfconfig.training_npz_dir.split(":")[1] + "*.npz")
        train_files = 400
        test_files = 100
    else:
        npz_list = tf.data.Dataset.list_files(tfconfig.training_npz_dir + "*.npz")
        train_files = 800
        test_files = 200
        val_files = 200
    # load data

    if tfconfig.data_mode == "new_old_mix":
        if tfconfig.use_TAPE_feature == 1:
            combined_dataset_pro = npz_list_pro.map(lambda x: tf.py_function(func=np_load, inp=[x],
                                                                   Tout=[tf.int16, tf.float32, tf.float32, tf.float32,
                                                                        tf.float32, tf.float32, tf.int16, tf.int32,
                                                                               tf.float32, tf.float32]))
            combined_dataset_rna = npz_list_rna.map(lambda x: tf.py_function(func=np_load, inp=[x],
                                                                    Tout=[tf.int16, tf.float32, tf.float32, tf.float32,
                                                                         tf.float32, tf.float32, tf.int16, tf.int32,
                                                                                   tf.float32, tf.float32]))
        else:
            combined_dataset_pro = npz_list_pro.map(lambda x: tf.py_function(func=np_load, inp=[x],
                                                                Tout=[tf.int16, tf.float32, tf.float32, tf.float32,
                                                                               tf.float32, tf.float32, tf.int16,
                                                                               tf.int32, tf.float32]))
            combined_dataset_rna = npz_list_rna.map(lambda x: tf.py_function(func=np_load, inp=[x],
                                                                         Tout=[tf.int16, tf.float32, tf.float32, tf.float32,
                                                                               tf.float32, tf.float32, tf.int16,
                                                                               tf.int32, tf.float32]))
    else:
        if tfconfig.use_TAPE_feature == 1:
            combined_dataset = npz_list.map(lambda x: tf.py_function(func=np_load, inp=[x],
                                                                 Tout=[tf.int16, tf.float32, tf.float32, tf.float32,
                                                                       tf.float32, tf.float32, tf.int16, tf.int32,
                                                                       tf.float32, tf.float32]))
        else:
            # return (proid_tf, rnatok_tf, pot_arr_hb_tf, pot_arr_pi_tf, pro_mask_tf, cross_mask_tf, label_tf,
            #         red_index_tf, protok_reduced,
            combined_dataset = npz_list.map(lambda x: tf.py_function(func=np_load, inp=[x],
                                                                     Tout=[tf.int16, tf.float32, tf.float32, tf.float32,
                                                                           tf.float32, tf.float32, tf.int16, tf.int32,
                                                                           tf.float32]))
    # batchfy
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    # batchfy, shard, prefetch
    if tfconfig.training == 1:
        dataset_batch = tfconfig.batch_size
        test_batch = tfconfig.test_batch_size
        # split into three sets
        if tfconfig.data_mode == "new_old_mix":
            combined_dataset_train_pro = combined_dataset_pro.take(int(train_files))
            combined_dataset_train_rna = combined_dataset_rna.take(int(train_files))
            combined_dataset_test_pro = combined_dataset_pro.skip(int(train_files)).take(int(test_files))
            combined_dataset_test_rna = combined_dataset_rna.skip(int(train_files)).take(int(test_files))
            combined_dataset_train = combined_dataset_train_pro.concatenate(combined_dataset_train_rna)
            combined_dataset_test = combined_dataset_test_pro.concatenate(combined_dataset_test_rna)
            combined_dataset_train = combined_dataset_train.repeat(1).shuffle(int(train_files)*2).batch(dataset_batch).with_options(options)
            combined_dataset_test = combined_dataset_test.repeat(1).shuffle(int(test_files)*2).batch(test_batch).with_options(options)


        else:
            combined_dataset_train = combined_dataset.take(int(train_files))
            combined_dataset_test = combined_dataset.skip(int(train_files)).take(int(test_files))
            combined_dataset_train = combined_dataset_train.repeat(1).shuffle(int(train_files)*2).batch(dataset_batch).with_options(options)
            combined_dataset_test = combined_dataset_test.repeat(1).shuffle(int(test_files)*2).batch(test_batch).with_options(options)
            # combined_dataset_train = combined_dataset_train.repeat(1).shuffle(int(train_files)).batch(dataset_batch, drop_remainder=True).with_options(options)
            # combined_dataset_test = combined_dataset_test.repeat(1).shuffle(int(train_files)).batch(test_batch, drop_remainder=True).with_options(options)
            # if tfconfig.data_mode != "all_unknown":
            #     combined_dataset_validation = combined_dataset.skip(int(train_files)).skip(test_files).take(int(val_files))
            #     combined_dataset_validation = combined_dataset_validation.repeat(1).batch(dataset_batch).with_options(
            #         options)
    else:
        dataset_batch = 1
        combined_dataset_test = combined_dataset.take(1)
        combined_dataset_test = combined_dataset_test.repeat(1).batch(dataset_batch, drop_remainder=True).with_options(options)
    # run model
    if tfconfig.training == 1:
        callbacks = [cp_callback_with_date, cp_callback_no_date, CustomCallback()]
    elif tfconfig.training == 0:
        callbacks = [CustomCallback()]

    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
    with strategy.scope():
        auc = tf.keras.metrics.AUC()
        loss_avg = tf.keras.metrics.Mean(name='train_loss')
        model = CustomModel(tfconfig)
        model.compile(optimizer=optimizer)
        if tfconfig.usechpoint == 1:
            model.load_weights(checkpoint_path_no_date)
    if tfconfig.training == 1:
        model.fit(combined_dataset_train, epochs=tfconfig.max_epoch, callbacks=callbacks, validation_data=combined_dataset_test)
    elif tfconfig.training == 0:
        model.evaluate(combined_dataset_test)


def flat_gradients(grads_or_idx_slices: tf.Tensor) -> tf.Tensor:
    if type(grads_or_idx_slices) == tf.IndexedSlices:
        return tf.scatter_nd(
            tf.expand_dims(grads_or_idx_slices.indices, 1),
            grads_or_idx_slices.values,
            tf.cast(grads_or_idx_slices.dense_shape, dtype="int32")
        )
    return grads_or_idx_slices


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    config = Tfconfig()

    #########################
    # Parser Setup
    #########################
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--group_to_ignore', type=int)
    parser.add_argument('--run_on_local', type=int)
    parser.add_argument('--node_name')
    parser.add_argument('--use_attn_augument', type=int)
    parser.add_argument('--clip_coeff', type=int)
    parser.add_argument('--two_d_softm', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--usechpoint', type=int)
    parser.add_argument('--init_lr', type=int)
    parser.add_argument('--self_player_num', type=int)
    parser.add_argument('--self_rlayer_num', type=int)
    parser.add_argument('--cross_layer_num', type=int)
    parser.add_argument('--rna_dff', type=int)
    parser.add_argument('--pro_dff', type=int)
    parser.add_argument('--cross_dff', type=int)
    parser.add_argument('--keyword')
    parser.add_argument('--training', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--max_epoch', type=int)
    parser.add_argument('--num_of_node', type=int)
    parser.add_argument('--roc_data_log', type=int)
    parser.add_argument('--max_pro_len', type=int)
    parser.add_argument('--aug_multiply', type=int, default=0)
    parser.add_argument('--data_dir_name')
    parser.add_argument('--two_d_softm_mul_row_count', type=int)
    parser.add_argument('--use_TAPE_feature', type=int)
    parser.add_argument('--only_rna_path', type=int)
    parser.add_argument('--only_protein_path', type=int)
    parser.add_argument('--data_mode', default="all")
    parser.add_argument('--cv_fold_id', type=int, default=None)
    parser.add_argument('--reduce_level', type=int)
    parser.add_argument('--reduced_pro_tokenlist', type=int)
    parser.add_argument('--test_batch_size', type=int)
    parser.add_argument('--test_freq', type=int)
    parser.add_argument('--dropout_rate', type=int)
    parser.add_argument('--initial_aug_coeff', type=int)

    #########################
    # put args to config
    #########################
    args = parser.parse_args()
    config.group_to_ignore = args.group_to_ignore
    config.data_dir_name = args.data_dir_name
    config.run_on_local = args.run_on_local
    config.node_name = args.node_name
    config.use_attn_augument = args.use_attn_augument
    config.two_d_softm = args.two_d_softm
    config.use_TAPE_feature = args.use_TAPE_feature
    config.num_heads = args.num_heads
    config.usechpoint = args.usechpoint
    config.init_lr = 10 ** (-1 * args.init_lr)
    config.self_player_num = args.self_player_num
    config.self_rlayer_num = args.self_rlayer_num
    config.cross_layer_num = args.cross_layer_num
    config.rna_dff = args.rna_dff
    config.pro_dff = args.pro_dff
    config.roc_data_log = args.roc_data_log
    config.cross_dff = args.cross_dff
    config.keyword = args.keyword
    config.training = args.training
    config.batch_size = args.batch_size
    config.test_batch_size = args.test_batch_size
    config.max_epoch = args.max_epoch
    config.num_of_node = args.num_of_node
    config.max_pro_len = args.max_pro_len
    config.clip_coeff = args.clip_coeff
    config.aug_multiply = args.aug_multiply
    config.only_rna_path = args.only_rna_path
    config.only_protein_path = args.only_protein_path
    config.data_mode = args.data_mode
    config.cv_fold_id = args.cv_fold_id
    config.reduce_level = args.reduce_level
    config.test_freq = args.test_freq
    config.initial_aug_coeff = args.initial_aug_coeff
    config.dropout_rate = args.dropout_rate * 0.1
    config.reduced_pro_tokenlist = args.reduced_pro_tokenlist
    config.two_d_softm_mul_row_count = args.two_d_softm_mul_row_count
    config.task_identifier = "node_" + str(config.node_name) + "_nodenum_" + str(config.num_of_node) + \
                             "_aug_" + str(config.use_attn_augument) + "_twoD_" + str(config.two_d_softm) + \
                             "_headnum_" + str(config.num_heads) + "_initLR_" + str(config.init_lr) + \
                             "_keywrd_" + str(config.keyword) + "_clip_coeff_" + str(config.clip_coeff) + \
                             "_cv_fold_id_" + str(config.cv_fold_id) + \
                             "_datamode_" + str(config.data_mode) + "_use_TAPE_feature_" + str(config.use_TAPE_feature)
    config = config.update(config.task_identifier)
    opt_trfmr(config)

