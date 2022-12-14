# -------------------------------------------------------------------
# Created on 2021.11.4
# Statistical potential table to augument attention
# was written, so now train with attention augumented
# with the potentials
# -------------------------------------------------------------------

# ------------------------------------------------
# Import
# ------------------------------------------------
import numpy as np
import tensorflow as tf
import os, socket
import datetime
import tensorflow_addons as tfa
import json, sys
sys.path.append("/gs/hs0/tga-science/kimura/transformer_tape_dnabert/anaconda/lib/python3.9/site-packages/")
import horovod.tensorflow as hvd

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session()

#########################
# Horovod, GPU Setup
#########################

seed = 0
tf.config.run_functions_eagerly(True)
tf.random.set_seed(seed)
operation_lebel_seed = 0
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
PROTEIN_MAX_LENGTH = 2805
DROPOUT_RATE = 0.1
final_target_vocab_size = 2
ATTN_OUT = "/gs/hs0/tga-science/kimura/transformer_tape_dnabert/AttentionAnalysis/output_weight/"

#########################
# Horovod, GPU Setup
#########################
hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.print(f"!!!!  gpus  {gpus}")
# tf.print(f"!!!!  hvd.local_rank()  {hvd.local_rank()}")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[hvd.local_rank()], enable=True)

seed = 0
tf.config.run_functions_eagerly(True)
tf.random.set_seed(seed)
operation_lebel_seed = 0
initializer = tf.keras.initializers.GlorotUniform(seed=operation_lebel_seed)

verbose = 1 if hvd.rank() == 0 else 0


class Tfconfig():
    def __init__(self):
        self.max_pro_len = PROTEIN_MAX_LENGTH
        self.max_rna_len = RNA_MAX_LENGTH
        self.num_layers = None
        self.d_model = D_MODEL
        self.num_heads = None
        self.dff = DFF
        self.dropout_rate = DROPOUT_RATE
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

    def update(self, task_id):
        # -------------
        # gpu number
        # -------------
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
            if self.data_mode == "unknown_broad_share":
                self.basepath = "/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/"
            else:
                self.basepath = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/"
            physical_devices = tf.config.list_physical_devices('CPU')  # 8 GPUs in my setup
            tf.config.set_visible_devices(physical_devices[0], 'CPU')  # Using all GPUs (default behaviour)
        elif self.run_on_local == 0:
            if self.data_mode == "unknown_broad_share":
                self.basepath = "/gs/hs0/tga-science/kimura/reduced_RBP_camas/"
            else:
                self.basepath = "/gs/hs0/tga-science/kimura/transformer_tape_dnabert/"
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
        # lnc or else
        # -------------
        if self.data_mode == "lnc":
            self.data_directory = "data_lncRNA"
        else:
            self.data_directory = "data"
        # -------------
        # input path
        # -------------
        # if self.data_mode != "all_unknown":
        #     self.training_npz_dir = f"{self.basepath}{self.data_directory}/training_data_tokenized/{self.data_dir_name}/"
        if self.data_dir_name == "all":
            self.training_npz_dir = f"{self.basepath}{self.data_directory}/training_data_tokenized/all5_small_tape/"
        elif self.data_dir_name[6] == "_" and self.data_dir_name[4] == "_":
            self.training_npz_dir = f"{self.basepath}{self.data_directory}/training_data_tokenized/attn_analysis_hb/{self.data_dir_name}/"
        elif self.data_dir_name == "debug":
            # /Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data_lncRNA/training_data_tokenized/
            self.training_npz_dir = f"{self.basepath}{self.data_directory}/training_data_tokenized/all5_small_tape/"
        else:
            # /Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data_lncRNA/training_data_tokenized/
            self.training_npz_dir = f"{self.basepath}{self.data_directory}/training_data_tokenized/{self.data_dir_name}/"
        if self.data_mode == "unknown_broad_share":
            self.training_npz_dir = f"{self.basepath}{self.data_directory}/training_data_tokenized_broad_share/"
        # -------------
        # chpoint path
        # -------------
        if self.training == 1:
            self.checkpoint_path = f"{self.basepath}chpoint/{self.keyword}/chpt_"
        elif self.training == 0:
            self.checkpoint_path = f"{self.basepath}chpoint/{self.keyword}/chpt_"
        self.checkpoint_dir = f"{self.basepath}chpoint/{self.keyword}/"

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


def scaled_dot_product_attention(k, q, v, tffig, one_on_rna, coeff_aug, coeff_att):

    ########################################################
    # Calculate Q x K
    ########################################################
    # one_on_rna = 1:rna-self, 0:protein-self, 2:cross
    matmul_qk = tf.cast(tf.matmul(q, k, transpose_b=True), tf.float32)  # (..., seq_len_q, seq_len_k)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    # assign smask in Decoder's 1st layer
    if one_on_rna == 1:  # rna self
        pass
    elif one_on_rna == 0:  # pro self
        mask = tffig.self_pro_mask_list
    else:            # cross
        mask = tffig.cross_padding_mask_list
    v = tf.cast(v, tf.float32)
    scaled_attention_logits = matmul_qk
    multiply_num = tf.cast(tffig.num_heads / 2, dtype="int32")

    ########################################################
    # add statistical potentials at Cross Attention Layers
    ########################################################
    if one_on_rna == 2 and tffig.use_attn_augument == 1:  # in a cross layer and the aug flag is on
        table_to_augment_hb = tffig.statpot_hb
        table_to_augment_pi = tffig.statpot_pi  # TensorShape([5, 2805, 101])
        table_to_augment = tf.transpose(tf.stack([table_to_augment_hb, table_to_augment_pi]), perm=[1, 0, 2, 3])  # [5, 2, 2805, 101]
        try:
            table_to_augment = tf.repeat(table_to_augment, repeats=[multiply_num, multiply_num], axis=1)  # [5, 4, 2805, 101]  THIS LINE IS WRONG!!!!
        except ValueError:
            table_to_augment = tf.repeat(table_to_augment, repeats=[multiply_num, multiply_num + 1], axis=1)
        if scaled_attention_logits.shape[-1] != table_to_augment.shape[-1]:
            table_to_augment = tf.transpose(table_to_augment, perm=[0, 1, 3, 2])
        # augment
        scaled_attention_logits *= coeff_att
        attn_weight_before = scaled_attention_logits
        if tffig.aug_multiply == 0:
            stats_to_add = coeff_aug * tf.cast(table_to_augment, dtype="float32")
            scaled_attention_logits += stats_to_add
        else:
            # table_to_augment *= tffig.aug_weight_aug
            scaled_attention_logits = \
                tf.math.multiply(tf.nn.softmax(tf.cast(table_to_augment, dtype="float32")), scaled_attention_logits)
    ########################################################
    # add large negative values
    ########################################################
    if one_on_rna != 1:  # except RNA
        if one_on_rna == 2:  # when cross, do transpose if needed
            if scaled_attention_logits.shape[-1] != mask.shape[-1]:
                mask = tf.transpose(mask, perm=[0, 2, 1])
        mask = tf.repeat(mask[:, tf.newaxis, :, :], repeats=tffig.num_heads, axis=1)
        scaled_attention_logits += (mask * -1e9)

    ########################################################
    # divide with squared root of dimension of leading vectors
    ########################################################
    scaled_attention_logits /= tf.math.sqrt(dk)

    ########################################################
    # apply softmax
    ########################################################
    if tffig.use_attn_augument == 1:
        if tffig.two_d_softm == 1:
            attention_weights = tf.cast(tf.keras.activations.softmax(scaled_attention_logits, axis=[2, 3]), tf.float32)
            # multiply the number of rows (necessary??)
            if tffig.two_d_softm_mul_row_count == 1:
                if attention_weights.shape[2] == tffig.max_pro_len:  # (5, 105, 2805)
                    attention_weights *= tf.cast(tf.math.count_nonzero(tffig.protok[0:1, :]), dtype="float32")
                elif attention_weights.shape[2] == 101:
                    attention_weights *= 101
        else:
            attention_weights = tf.cast(tf.keras.activations.softmax(scaled_attention_logits, axis=3), tf.float32)
    else:
        attention_weights = tf.cast(tf.keras.activations.softmax(scaled_attention_logits, axis=3), tf.float32)

    ########################################################
    # calculate weight matrix x V
    ########################################################
    output = tf.matmul(tf.transpose(attention_weights, perm=[0, 1, 3, 2]), v)  # (..., seq_len_q, depth_v)

    ########################################################
    # pack matrices of cross attention
    ########################################################
    if one_on_rna == 2 and tffig.use_attn_augument == 1:
        attention_weights = [attn_weight_before, stats_to_add, attention_weights]

    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu', kernel_initializer=initializer),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model, kernel_initializer=initializer)  # (batch_size, seq_len, d_model)
    ])


class FinalLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(FinalLayer, self).__init__()
        self.finallayer = tf.keras.layers.Dense(final_target_vocab_size, activation='softmax',
                                                kernel_initializer=initializer)

    def call(self, x1, x2, cffg):   # x1 : rna, x2 : pro
        if cffg.only_protein_path == 1:
            x = tf.reduce_mean(x2, axis=1)
        elif cffg.only_rna_path == 1:
            x = tf.reduce_mean(x1, axis=1)
        else:
            x = tf.keras.layers.Average()([tf.reduce_mean(x1, axis=1), tf.reduce_mean(x2, axis=1)])
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


class Embedders(tf.keras.layers.Layer):
    def __init__(self, encfg, one_if_rna):
        super(Embedders, self).__init__()
        if one_if_rna == 1:
            vocab_size = 4
            maxlen = encfg.max_rna_len
            self.embedding = tf.keras.layers.Embedding(vocab_size + 1, encfg.d_model, mask_zero=True)  # 5 because 4 bases and an unknown
        elif one_if_rna == 0:
            maxlen = encfg.max_pro_len
            vocab_size = 20
            if encfg.use_TAPE_feature == 1:
                self.dense = tf.keras.layers.Dense(encfg.d_model, kernel_initializer=initializer)
            else:
                self.embedding = tf.keras.layers.Embedding(vocab_size + 1, encfg.d_model, mask_zero=True)  # 5 because 4 bases and an unknown
        self.d_model = tf.cast(encfg.d_model, tf.float32)
        self.pos_encoding = positional_encoding(maxlen, encfg.d_model)
        self.dropout = tf.keras.layers.Dropout(encfg.dropout_rate)

    def call(self, tf_cfg, one_when_rna):  # tfcfg, RCODES, i, 1, self.rna_out
        if one_when_rna == 1:
            sequence = tf_cfg.rnatok
            maxlen = tf_cfg.max_rna_len
            x = self.embedding(sequence)
        elif one_when_rna == 0:
            sequence = tf_cfg.protok
            tf.print(f"tf_cfg.p_tape_tf.shape {tf_cfg.p_tape_tf.shape}")  # sequence.shape (1, 5, 2805)
            maxlen = tf_cfg.max_pro_len
            # using TAPE output
            if tf_cfg.use_TAPE_feature == 1:
                x = self.dense(tf_cfg.p_tape_tf)
            # embedding
            else:
                x = self.embedding(sequence)
                x = x[0]
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # x=(1, 662, 64)
        # padding
        if one_when_rna == 0:  # protein
            x += self.pos_encoding[:maxlen, :]
        else:  # RNA
            x += self.pos_encoding[:maxlen, :]
        # dropout
        x = self.dropout(x, training=tf_cfg.training_boolean)
        x = tf.cast(x, tf.float32)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x /= tf.cast(self.d_model, tf.float32)

        # change x to reduced sequence with reduced_index
        # need to add zero padding to
        #
        #

        return x


class Encoders(tf.keras.layers.Layer):
    def __init__(self, trscfg, one_if_rna):
        super(Encoders, self).__init__()
        if one_if_rna == 1:
            self.layer_num = trscfg.self_rlayer_num
        else:
            self.layer_num = trscfg.self_player_num
        for i in range(self.layer_num):
            self.rencoder = SelfAttention(trscfg, one_if_rna)
    def call(self, tfcfg, x):
        weights = {}
        for i in range(self.layer_num):
            x, weights[f"layer_{i+1}"] = self.rencoder.call(tfcfg, x)
        return x, weights


class CrossLayers(tf.keras.layers.Layer):
    def __init__(self, trscfg):
        super(CrossLayers, self).__init__()
        for i in range(trscfg.cross_layer_num):
            self.cross_attention = CrossAttention(trscfg)
    def call(self, tfcfg, rna_out, pro_out):
        pro_cross_weights, rna_cross_weights = {}, {}
        for i in range(tfcfg.cross_layer_num):
            pro_out, rna_out, pro_cross_weights[f"layer_{i+1}"], rna_cross_weights[f"layer_{i+1}"] \
                = self.cross_attention.call(tfcfg, pro_out, rna_out)  # q, kv, confg, layr_num, one_when_rna
        return pro_out, rna_out, pro_cross_weights, rna_cross_weights


class CrossAttention(tf.keras.layers.Layer):  # two inputs/outputs
    def __init__(self, cr_cfg):
        super(CrossAttention, self).__init__()
        if cr_cfg.only_rna_path == 0:
            self.rna_cross_attention = AttentionLayer(cr_cfg, cr_cfg.cross_dff)
        else:
            pass
        if cr_cfg.only_protein_path == 0:
            self.pro_cross_attention = AttentionLayer(cr_cfg, cr_cfg.cross_dff)
        else:
            pass
    def call(self, cross_cfg, pro_inp, rna_inp):
        # q, kv, confg, layr_num, one_when_rna
        if cross_cfg.only_rna_path == 0:
            rna_out, rna_cross_weights = self.rna_cross_attention.call(rna_inp, pro_inp, cross_cfg, 2)  # q, kv
        else:
            rna_out = rna_inp
            rna_cross_weights = {}
        if cross_cfg.only_protein_path == 0:
            pro_out, pro_cross_weights = self.pro_cross_attention.call(pro_inp, rna_inp, cross_cfg, 2)
        else:
            pro_out = pro_inp
            pro_cross_weights = {}
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
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=confg.training_boolean)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
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

        # define coeffs for augmentation
        if mhacfg.use_attn_augument == 1:
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
                initializer=tf.constant_initializer(1))
            if mhacfg.clip_coeff == 1:
                self.aug_weight_aug = tf.sigmoid(self.aug_weight_aug)
                self.aug_weight_att = tf.sigmoid(self.aug_weight_att)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        # x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, kv, tfcg, one_for_rna):
        batch_size = tfcg.batch_size
        v = self.wv(kv)
        k = self.wk(kv)
        q = self.wq(q)
        # split
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        # calc. attention
        # tf.print(f"coeff att {self.aug_weight_att.numpy()}, coeff aug {self.aug_weight_aug.numpy()}")
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, tfcg, one_for_rna, self.aug_weight_aug, self.aug_weight_att)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
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
    # transformer = Transformer(tfconfig)
    learning_rate = tfconfig.init_lr
    # train_step_signature = [
    #     tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    #     tf.TensorSpec(shape=(None, None), dtype=tf.int32),]

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
        def train_step(self, data_combined):
            tf.print(f"### {socket.gethostname()} in python script ###")

            datalist = [x for x in data_combined]
            tfconfig.proid = self.flatten_n_batch(datalist[0])
            tfconfig.protok = self.flatten_n_batch(datalist[1])
            tfconfig.rnatok = self.flatten_n_batch(datalist[2])
            tfconfig.statpot_hb = self.flatten_n_batch(datalist[3])
            tfconfig.statpot_pi = self.flatten_n_batch(datalist[4])
            tfconfig.self_pro_mask_list = self.flatten_n_batch(datalist[5])
            tfconfig.cross_padding_mask_list = self.flatten_n_batch(datalist[6])
            tfconfig.label = self.flatten_n_batch(datalist[7])

            if tfconfig.use_TAPE_feature == 1:
                tfconfig.p_tape_tf = self.flatten_n_batch(datalist[8])
            with tf.GradientTape() as tape:
                predictions = self.transformer(tfconfig)  # prediction, pweight, rweight, p_cr_weight, r_cr_weight
                loss = loss_function(tfconfig.label[0], predictions)
            tape = hvd.DistributedGradientTape(tape)

            if tfconfig.roc_data_log == 1:
                tf.print(f"### {socket.gethostname()} ###train ROC_DATA:" + ":" + str(predictions.numpy()).replace('\n', '') + "-" + str(tfconfig.label).replace('\n', ''))
            loss_avg.update_state(loss)
            gradients = tape.gradient(loss, self.trainable_variables)
            auc.update_state(tfconfig.label[0], predictions)
            tfconfig.losssum += np.sum(loss.numpy())
            if tfconfig.training == 1:
                trainable_variables = [var for grad, var in zip(gradients, self.transformer.trainable_variables) if
                                       grad is not None]
                gradient = [flat_gradients(grad) for grad in gradients if grad is not None]
                # gradient = tf.distribute.get_replica_context().all_reduce(
                #     'sum', gradient, options=options)
                optimizer.apply_gradients(zip(gradient, trainable_variables))

                # if first_batch:
                #     hvd.broadcast_variables(self.trainable_variables, root_rank=0)
                #     hvd.broadcast_variables(optimizer.trainable_variables, root_rank=0)

            tfconfig.batch_count += 1
            return {"loss": loss_avg.result(), "auc": auc.result()}


        @tf.function()
        # @tf.function(input_signature=train_step_signature)
        def test_step(self, data_combined, y=None, first_batch=None):
            datalist = [x for x in data_combined]
            tfconfig.proid = self.flatten_n_batch(datalist[0])
            tfconfig.protok = self.flatten_n_batch(datalist[1])
            tfconfig.rnatok = self.flatten_n_batch(datalist[2])
            tfconfig.statpot_hb = self.flatten_n_batch(datalist[3])
            tfconfig.statpot_pi = self.flatten_n_batch(datalist[4])
            tfconfig.self_pro_mask_list = self.flatten_n_batch(datalist[5])
            tfconfig.cross_padding_mask_list = self.flatten_n_batch(datalist[6])
            tfconfig.label = self.flatten_n_batch(datalist[7])
            if tfconfig.use_TAPE_feature == 1:
                tfconfig.p_tape_tf = self.flatten_n_batch(datalist[8])

            if tfconfig.run_attn_analysis == 1:
                predictions, pweight, rweight, pself, rself, proout, rnaout = self.transformer(tfconfig)  # prediction, p_cr_weight, r_cr_weight, pweight, rweight, pout, rout
            else:
                predictions = self.transformer(tfconfig)  # prediction, pweight, rweight, p_cr_weight, r_cr_weight
            loss = loss_function(tfconfig.label[0], predictions)
            loss_avg.update_state(loss)
            auc.update_state(tfconfig.label[0], predictions)
            tfconfig.batch_count += 1
            if tfconfig.run_attn_analysis == 1:
                if not os.path.exists(f"{ATTN_OUT}/{tfconfig.data_dir_name}"):
                    os.mkdir(f"{ATTN_OUT}/{tfconfig.data_dir_name}")
                time_and_task = f"_{tfconfig.usechpoint}_{tfconfig.keyword}"
                # -------------------------
                # write final output
                # -------------------------
                np.save(f"{ATTN_OUT}/{tfconfig.data_dir_name}/attn_analysis_hb_proout{time_and_task}", proout)
                np.save(f"{ATTN_OUT}/{tfconfig.data_dir_name}/attn_analysis_hb_RNAout{time_and_task}", rnaout)

                # -------------------------
                # write FINAL cross weight
                # -------------------------
                tf.print(np.save(f"{ATTN_OUT}/{tfconfig.data_dir_name}/final_cross_pro{time_and_task}", np.array(pweight["layer_1"][2])))
                tf.print(np.save(f"{ATTN_OUT}/{tfconfig.data_dir_name}/final_cross_rna{time_and_task}", np.array(rweight["layer_1"][2])))

                # -------------------------
                # write added pot matrix at cross
                # -------------------------
                tf.print(np.save(f"{ATTN_OUT}/{tfconfig.data_dir_name}/added_pot_cross_pro{time_and_task}", np.array(pweight["layer_1"][1])))
                tf.print(np.save(f"{ATTN_OUT}/{tfconfig.data_dir_name}/added_pot_cross_rna{time_and_task}", np.array(rweight["layer_1"][1])))

                # -------------------------
                # write FIRST cross weight
                # -------------------------
                tf.print(np.save(f"{ATTN_OUT}/{tfconfig.data_dir_name}/first_cross_pro{time_and_task}", np.array(pweight["layer_1"][0])))
                tf.print(np.save(f"{ATTN_OUT}/{tfconfig.data_dir_name}/first_cross_rna{time_and_task}", np.array(rweight["layer_1"][0])))

                # for i in range(1):
                #     # -------------------------
                #     # write self attention weight
                #     # -------------------------
                #     np.save(f"{ATTN_OUT}/{tfconfig.data_dir_name}/attn_analysis_hb_pro_self_l_{i+1}", np.array(pself[f"layer_{i+1}"]))
                #     np.save(f"{ATTN_OUT}/{tfconfig.data_dir_name}/attn_analysis_hb_rna_self_l_{i+1}", np.array(rself[f"layer_{i+1}"]))
            if tfconfig.roc_data_log == 1:
                tf.print("test ROC_DATA:" + ":" + str(predictions.numpy()).replace('\n', '') + "-" + str(tfconfig.label).replace('\n', ''))
            return {"test_auc": auc.result(), "test_loss": loss_avg.result()}

        @property
        def metrics(self):
            loss_avg = tf.keras.metrics.Mean(name='train_loss')
            auc = tf.keras.metrics.AUC()
            return [loss_avg, auc]


    def get_unknown_npzlist(fold_id):  # return tf.dataset consisting of 800 training and 200 test
        # /transformer_tape_dnabert/data_lncRNA/training_data_tokenized/0
        four_dataset = None
        five_dataset = None
        first_done = 0
        # add training set four times
        for i in range(5):
            groupdirpath = f"{tfconfig.training_npz_dir}{i}/"
            if first_done == 0:
                if i != fold_id:
                    four_dataset = tf.data.Dataset.list_files(groupdirpath + "*.npz")
                    first_done = 1
                else:
                    pass
            else:
                four_dataset = four_dataset.concatenate(tf.data.Dataset.list_files(groupdirpath + "*.npz"))
        # add test set at the end
        five_dataset = four_dataset.concatenate(tf.data.Dataset.list_files(f"{tfconfig.training_npz_dir}{fold_id}/" + "*.npz"))
        return five_dataset


    def np_load(filename):
        arr = np.load(filename.numpy(), allow_pickle=True)
        proid_tf = tf.convert_to_tensor(arr["proid"], dtype="int16")
        protok_tf = tf.convert_to_tensor(arr["protok"], dtype="float32")
        rnatok_tf = tf.convert_to_tensor(arr["rnatok"], dtype="float32")
        pro_mask_tf = tf.convert_to_tensor(arr["pro_mask"], dtype="float32")
        cross_mask_tf = tf.convert_to_tensor(arr["cross_mask"], dtype="float32")
        pot_arr_hb_tf = tf.convert_to_tensor(arr["hb_pots"], dtype="float32")
        pot_arr_pi_tf = tf.convert_to_tensor(arr["pi_pots"], dtype="float32")
        label_tf = tf.convert_to_tensor(arr["label"], dtype="int16")
        if tfconfig.use_TAPE_feature == 1:
            p_tape_tf = tf.convert_to_tensor(arr["p_tape_arr"], dtype="float32")
            return (proid_tf, protok_tf, rnatok_tf, pot_arr_hb_tf, pot_arr_pi_tf, pro_mask_tf, cross_mask_tf, label_tf, p_tape_tf)
        else:
            return (proid_tf, protok_tf, rnatok_tf, pot_arr_hb_tf, pot_arr_pi_tf, pro_mask_tf, cross_mask_tf, label_tf)


    # ------------------------------------------------
    # callbacks and dataset
    # ------------------------------------------------

    optimizer = tfa.optimizers.RectifiedAdam(learning_rate * hvd.size(), beta_1=0.9, beta_2=0.98, epsilon=1e-9,
                                             # weight_decay=1e-5)
                                             weight_decay=1e-5, clipvalue=1)
    current_time = datetime.datetime.now().strftime("%Y%m%d")
    checkpoint_path = f"{tfconfig.checkpoint_path}" if hvd.rank() == 0 else None
    # checkpoint_path = f"./chpoint_{tfconfig.use_TAPE_feature}{tfconfig.use_attn_augument}{tfconfig.clip_coeff}{tfconfig.two_d_softm}"
    if not os.path.exists(tfconfig.checkpoint_dir):
        os.mkdir(tfconfig.checkpoint_dir)

    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, log=None, log2=None):
            auc.reset_states()
            loss_avg.reset_states()

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     monitor="eval_loss",
                                                     mode="min",
                                                     save_weights_only=True,
                                                     verbose=1)
    # get paths
    if tfconfig.data_mode == "all_unknown_broad_share":
        npz_list = get_unknown_npzlist(tfconfig.cv_fold_id)
        train_files = 2400
        test_files = 600
        val_files = 0
    elif tfconfig.data_mode == "all_unknown":
        npz_list = get_unknown_npzlist(tfconfig.cv_fold_id)
        train_files = 800
        test_files = 200
        val_files = 0
    elif tfconfig.data_mode == "debug":
        npz_list = tf.data.Dataset.list_files(tfconfig.training_npz_dir + "*.npz")
        train_files = 4
        test_files = 4
        val_files = 0
    else:
        npz_list = tf.data.Dataset.list_files(tfconfig.training_npz_dir + "*.npz")
        train_files = 1000
        test_files = 200
        val_files = 200
    # load data
    if tfconfig.use_TAPE_feature == 1:
        combined_dataset = npz_list.map(lambda x: tf.py_function(func=np_load, inp=[x],
            Tout=[tf.int16, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int16, tf.float32]))
    else:
        combined_dataset = npz_list.map(lambda x: tf.py_function(func=np_load, inp=[x],
            Tout=[tf.int16, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int16]))

    # ------------------------------------------------
    # Collective Policy
    # ------------------------------------------------
    options = tf.data.Options()
    # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    # batchfy, shard, prefetch
    # dataset_batch = int(tfconfig.num_of_gpu * (tfconfig.batch_size / 5))
    if tfconfig.training == 1:
        dataset_batch = 1
        # dataset_batch = 4 * config.num_of_node
        tf.print(f"dataset_batch {dataset_batch}")
        # split into three sets
        combined_dataset_train = combined_dataset.take(int(train_files))
        combined_dataset_train = combined_dataset_train.repeat(1).batch(dataset_batch)
        # combined_dataset_train = combined_dataset_train.repeat(1).batch(dataset_batch).with_options(options)
        combined_dataset_test = combined_dataset.skip(int(train_files)).take(int(test_files))
        combined_dataset_test = combined_dataset_test.repeat(1).batch(dataset_batch)
        # combined_dataset_test = combined_dataset_test.repeat(1).batch(dataset_batch).with_options(options)
        if tfconfig.data_mode != "all_unknown":
            combined_dataset_validation = combined_dataset.skip(int(train_files)).skip(test_files).take(int(val_files))
            combined_dataset_validation = combined_dataset_validation.repeat(1).batch(dataset_batch).with_options(options)

        # # distribute the tf.dataset
        # combined_dataset_train = strategy.experimental_distribute_dataset(combined_dataset_train)
        # combined_dataset_test = strategy.experimental_distribute_dataset(combined_dataset_test)
    else:
        dataset_batch = 1
        combined_dataset_test = combined_dataset.take(1)
        combined_dataset_test = combined_dataset_test.repeat(1).batch(dataset_batch)
        # combined_dataset_test = combined_dataset_test.repeat(1).batch(dataset_batch).with_options(options)
        # distribute the tf.dataset
        # combined_dataset_test = strategy.experimental_distribute_dataset(combined_dataset_test)
    # run model
    auc = tf.keras.metrics.AUC()
    loss_avg = tf.keras.metrics.Mean(name='train_loss')
    if tfconfig.training == 1:
        callbacks = [cp_callback, CustomCallback()]
    elif tfconfig.training == 0:
        callbacks = [CustomCallback()]

    # with strategy.scope():
    model = CustomModel(tfconfig)
    model.compile(optimizer=optimizer)
    if tfconfig.usechpoint == 1 and hvd.rank() == 0:
        model.load_weights(tfconfig.checkpoint_path)
    if tfconfig.training == 1:
        model.fit(combined_dataset_train, epochs=tfconfig.max_epoch,
                  callbacks=callbacks,
                  validation_data=combined_dataset_test)
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
    from subprocess import call
    # call(["unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY"], shell=True)

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
    parser.add_argument('--num_accum_grad', type=int)
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
    parser.add_argument('--datafiles_num', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--max_epoch', type=int)
    parser.add_argument('--num_of_node', type=int)
    parser.add_argument('--roc_data_log', type=int)
    parser.add_argument('--max_pro_len', type=int)
    parser.add_argument('--aug_multiply', type=int, default=0)
    parser.add_argument('--task_identifier')
    parser.add_argument('--data_dir_name')
    # parser.add_argument('--hostnames')
    parser.add_argument('--two_d_softm_mul_row_count', type=int)
    parser.add_argument('--use_TAPE_feature', type=int)
    parser.add_argument('--only_rna_path', type=int)
    parser.add_argument('--only_protein_path', type=int)
    parser.add_argument('--data_mode', default="all")
    parser.add_argument('--cv_fold_id', type=int, default=None)
    parser.add_argument('--run_attn_analysis', type=int, default=0)

    #########################
    # put args to config
    #########################
    args = parser.parse_args()
    config.group_to_ignore = args.group_to_ignore
    config.data_dir_name = args.data_dir_name
    config.run_on_local = args.run_on_local
    config.node_name = args.node_name
    config.use_attn_augument = args.use_attn_augument
    config.num_accum_grad = args.num_accum_grad
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
    config.datafiles_num = args.datafiles_num
    config.batch_size = args.batch_size
    config.max_epoch = args.max_epoch
    config.num_of_node = args.num_of_node
    config.max_pro_len = args.max_pro_len
    config.clip_coeff = args.clip_coeff
    config.aug_multiply = args.aug_multiply
    config.only_rna_path = args.only_rna_path
    config.only_protein_path = args.only_protein_path
    config.data_mode = args.data_mode
    config.cv_fold_id = args.cv_fold_id
    # config.hostnames = args.hostnames
    config.run_attn_analysis = args.run_attn_analysis
    config.two_d_softm_mul_row_count = args.two_d_softm_mul_row_count
    config.task_identifier = "node_" + str(config.node_name) + "_nodenum_" + str(config.num_of_node) +  \
                            "_aug_" + str(config.use_attn_augument) + "_twoD_" + str(config.two_d_softm) +\
                            "_headnum_" + str(config.num_heads) + "_initLR_" + str(config.init_lr) +\
                             "_keywrd_" + str(config.keyword) + "_clip_coeff_" + str(config.clip_coeff) +\
                            "_cv_fold_id_" + str(config.cv_fold_id) + \
                            "_datamode_" + str(config.data_mode) + "_use_TAPE_feature_" + str(config.use_TAPE_feature)
    config = config.update(config.task_identifier)

    opt_trfmr(config)
