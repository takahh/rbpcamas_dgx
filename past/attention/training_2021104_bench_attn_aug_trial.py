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
import os
import datetime
import sys
import time

tf.config.run_functions_eagerly(True)
# tf.enable_eager_execution()
# np.set_printoptions(threshold=sys.maxsize)
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# gpu_options = tf.GPUOptions(allow_growth=True)

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------


def make_pro_dic(path, key):
    mask_dict = {}
    for files in os.listdir(path):
        if "npy" in files:
            protname = files.split(".")[0]
            arrz = np.load(f"{path}{files}", allow_pickle=True)
            mask_dict[protname] = arrz[key]
    return mask_dict


########################################################################
# transformer
# variables to consider
########################################################################
# BASE_PATH = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/"
BASE_PATH = "/gs/hs0/tga-science/kimura/transformer_tape_dnabert/"
#
# protein_seq_dict_padded = make_pro_dic(PROTEIN_SEQ_FILE, "padded_array")
# protein_seq_dict_unpadded = make_pro_dic(PROTEIN_SEQ_FILE, "unpadded_array")
# cross_mask_dict = make_pro_dic(PROTEIN_SEQ_FILE, "cross_mask")
# cross_mask_small_dict = make_pro_dic(PROTEIN_SEQ_FILE, "cross_mask_small")
WARMUP_STEPS = 5
USE_CHECKPOINT = False
MAX_EPOCHS = 10000
D_MODEL = 64
DFF = 64
DROPOUT_RATE = 0.1
final_target_vocab_size = 2
########################################################################
########################################################################

path = BASE_PATH


class Tfconfig():
    def __init__(self):
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

    def update(self):
        if int(self.use_attn_augument) == 1:  # num to string for task name
            self.aug_or_not = "aug"
        else:
            self.aug_or_not = "noaug"
        self.pairfile = f"{BASE_PATH}data/benchmarks/label/{self.benchname}_pairs_shuffled.txt"
        self.rna_seq_file = f"{BASE_PATH}data/benchmarks/sequence/{self.benchname}_rna_seq.fa "  # AARS.npy
        self.protein_seq_file = f"{BASE_PATH}data/benchmarks/sequence/{self.benchname}_protein_seq.fa "  # 0.npy
        self.protein_feature_file = f"{BASE_PATH}data/benchmark768_protein/{self.benchname}/"  # AARS.npy
        self.rna_feature_file = f"{BASE_PATH}data/benchmark768_RNA/{self.benchname}/"  # 0.npy
        # "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/finetune_bench/RPI369"
        self.cv_list_file = f"{BASE_PATH}data/benchmarks/id_list_{self.benchname}_for_5CV.csv.npz"
        self.statistical_pot_path = f"{BASE_PATH}data/attn_arrays/{self.benchname}/"  # 1a1t-A_1a1t-B.npz
        self.statistical_pot_pi_path = f"{BASE_PATH}data/attn_arrays_pi/{self.benchname}/"  # 1a1t-A_1a1t-B.npz
        self.taskname = f"_{self.benchname}_cv{self.group_to_ignore}_{self.aug_or_not}_lr{self.init_lr}"
        self.checkpoint_path = f"{BASE_PATH}_{self.taskname}"


# ------------------------------------------------
# Functions
# ------------------------------------------------


def get_stat_pot_arr(tfcfg):
    arr = np.load(f"{tfcfg.statistical_pot_path}{tfcfg.pair_name}.npz", allow_pickle=True)
    return arr["data"]


def get_pi_stat_pot_arr(tfcfg2):
    arr = np.load(f"{tfcfg2.statistical_pot_pi_path}{tfcfg2.pair_name}.npz", allow_pickle=True)
    return arr["data"]


def scaled_dot_product_attention(k, q, v, location, tffig, block_num=None, augweight_hb=None, augweight_pi=None, augweight_at=None):
    # location = 1:dec1(self), 2:dec2(cross), 3:enc(self)
    # mask depends on location
    # 1st Decoder1: self_pro_mask_list  loc 1, block 0
    # 1st Decoder2: cross_padding                loc 2, block 0
    # 2~  Decoder1: cross_small                  loc 1
    # 2~  Decoder2: cross_small                  loc 2
    # Encoder : loc 3, self_rna_mask_list

    # calculate attention table
    matmul_qk = tf.cast(tf.matmul(q, k, transpose_b=True), tf.float32)  # (..., seq_len_q, seq_len_k)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    # assign smask in Decoder's 1st layer

    if location == 3:  # encoder
        mask = tffig.self_rna_mask_list
    elif location == 1:  # decoder 1st cell
        mask = tffig.self_pro_mask_list
    else:            # decoder 2nd cell
        mask = tffig.cross_padding_mask_list

    v = tf.cast(v, tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  # (1, 34, 29, 29)
    scaled_attention_logits += (mask * -1e9)
    # attention_weights = tf.cast(tf.nn.softmax(scaled_attention_logits, axis=-1), tf.float32)
    # attention_weights = tf.cast(tf.nn.softmax(scaled_attention_logits, axis=-1), tf.float32)

    ########################################################
    # add astatistical potentials Decoder second Cell
    ########################################################
    if location == 2 and tffig.use_attn_augument == 1:
        table_to_augment = get_stat_pot_arr(tffig)
        table_to_augment_pi = get_pi_stat_pot_arr(tffig)
        scaled_attention_logits *= augweight_at[block_num]
        scaled_attention_logits += augweight_hb[block_num] * table_to_augment
        scaled_attention_logits += augweight_pi[block_num] * table_to_augment_pi
        attention_weights = tf.cast(tf.keras.activations.softmax(scaled_attention_logits, axis=[2, 3]), tf.float32)
    else:
        attention_weights = tf.keras.activations.softmax(scaled_attention_logits, axis=[2, 3])

    # prob has shape [BS, H, W, CH]
    # tf.print(tf.reduce_sum(attention_weights, axis=[2, 3]))

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    # output = tf.matmul(tf.transpose(attention_weights, perm=[0, 1, 3, 2]), v)  # (..., seq_len_q, depth_v)
    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class FinalLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(FinalLayer, self).__init__()
        self.finallayer = tf.keras.layers.Dense(final_target_vocab_size, activation='softmax')
        # self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):  # x is a 2d vector
        # normalize
        x = self.finallayer(x)  # [[-0.953382075 0.301765621]]
        # logits = self.layernorm(x)
        # logits = tf.math.l2_normalize(self.finallayer(x))  # [[-0.953382075 0.301765621]]
        # softmax
        # finalvalue = tf.nn.softmax(logits, axis=-1)
        # return finalvalue
        return x


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, trscfg, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, trscfg, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, trscfg, rate)
        self.final_layer = FinalLayer()

    def call(self, training, tfcfg):
        tfcfg.enc_output = self.encoder.call(training, tfcfg)  # (batch_size, inp_seq_len, d_model)
        dec_output, attention_weights = self.decoder.call(training, tfcfg)
        final_output = self.final_layer.call(dec_output[:, 0, :])  # output: 1-dimensional values
        # final_output = self.final_layer.call(tf.reduce_sum(dec_output, axis=-2))  # output: 1-dimensional values
        return final_output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, num_layers, mhacfg):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads  # depth = 128 /4 = 32
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)
        if mhacfg.use_attn_augument == 1:
            self.aug_weight_hb = tf.compat.v1.get_variable(
                "aug_weight_hb",
                shape=[num_layers, 1],
                trainable=True,
                dtype="float32",
                initializer=tf.random_uniform_initializer(maxval=1, minval=0),
                constraint=lambda t: tf.clip_by_value(t, 0, 1))
            self.aug_weight_pi = tf.compat.v1.get_variable(
                "aug_weight_pi",
                shape=[num_layers, 1],
                trainable=True,
                dtype="float32",
                initializer=tf.random_uniform_initializer(maxval=1, minval=0),
                constraint=lambda t: tf.clip_by_value(t, 0, 1))
            self.aug_weight_at = tf.compat.v1.get_variable(
                "aug_weight_at",
                shape=[num_layers, 1],
                trainable=True,
                dtype="float32",
                initializer=tf.random_uniform_initializer(maxval=1, minval=0),
                constraint=lambda t: tf.clip_by_value(t, 0, 1))

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, q, k, tfcg, block_num, mha_num):
        batch_size = tf.shape(q)[0]
        v = self.wv(v)
        k = self.wk(k)
        q = self.wq(q)
        # split
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        # calc. attention
        if tfcg.use_attn_augument == 1:
            scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mha_num, tfcg, block_num,
                                                            self.aug_weight_hb, self.aug_weight_pi, self.aug_weight_at)
        else:
            scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mha_num, tfcg, block_num)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, num_layers, elcfg, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, num_layers, elcfg)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, confg, block_num=None):
        attn_output, _ = self.mha.call(x, x, x, confg, block_num, 3)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, num_layers, dlcfg, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads, num_layers, dlcfg)
        self.mha2 = MultiHeadAttention(d_model, num_heads, num_layers, dlcfg)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-3)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-3)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-3)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

        # x, cofg.enc_output, training, cfg.cross_padding_mask_list, tfconfig, cfg.pro_padding_mask, i, cfg.cross_padding_mask_list_small
    def call(self, x, cofg, training, block_num=None):
        # self attn layer
        attn1, attn_weights_block1 = self.mha1(x, x, x, cofg, block_num, 1)
        attn1 = self.dropout1(attn1, training=training)  # (49, 105, 768)
        # tf.print(f"attn.shape is {attn1.shape}, x.shape is {x.shape}, block_num is {block_num}")
        out1 = self.layernorm1(attn1 + x)

        # Cross attn layer                       v               q                 k
        attn2, attn_weights_block2 = self.mha2(cofg.enc_output, cofg.enc_output, out1, cofg, block_num, 2)
        attn2 = self.dropout2(attn2, training=training)  # (49, 105, 768)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
        # return out3, attn_weights_block1, attn_weights_block2
        return out3, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, encfg, rate=0.1):
        super(Encoder, self).__init__()
        self.first = tf.keras.layers.Dense(d_model, activation='relu')
        self.d_model = tf.cast(d_model, tf.float32)
        self.num_layers = num_layers
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, num_layers, encfg, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, training, tf_cfg):
        x = self.first(tf_cfg.rna_vec_list)
        x = self.dropout(x, training=training)
        x = tf.cast(x, tf.float32)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x /= tf.cast(self.d_model, tf.float32)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, tf_cfg, i)
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, decfg, rate=0.1):
        super(Decoder, self).__init__()
        self.first = tf.keras.layers.Dense(d_model, activation='relu')
        self.d_model = d_model
        self.num_layers = num_layers
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, num_layers, decfg, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, training, cfg):  # x = tar previously
        attention_weights = {}
        x = self.first(cfg.pvecs_list_padded)
        x = tf.cast(x, tf.float32)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x /= tf.cast(self.d_model, tf.float32)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block1 = self.dec_layers[i](x, cfg, training, i)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1

        return x, attention_weights


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=WARMUP_STEPS):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class CustomSchedule2(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=WARMUP_STEPS):
        super(CustomSchedule2, self).__init__()

    def __call__(self, step):
        rate = 2 ** (-step * 10)
        return rate


def opt_trfmr(tfconfig):

    # ------------------------------------------------
    # Constant
    # ------------------------------------------------
    use_checkpoint = USE_CHECKPOINT
    num_layers = tfconfig.num_layers
    d_model = tfconfig.d_model
    dff = tfconfig.dff
    num_heads = tfconfig.num_heads
    dropout_rate = tfconfig.dropout_rate

    max_epoch = tfconfig.max_epoch
    checkpoint_path = tfconfig.checkpoint_path
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)


    def accuracy_function(labellist, predictions):
        predicted_labels = tf.math.greater_equal(predictions, tf.constant(0.5))
        predicted_labels = tf.cast(predicted_labels, dtype="int32")
        accuracies = tf.math.equal(predicted_labels, labellist[0])
        accuracies = tf.cast(accuracies, dtype="float32")
        return tf.reduce_sum(accuracies) / tf.constant(200, dtype="float32")


    def loss_function(labellist, predictions):
        if labellist[0] == 1:
            new_labellist = [0, 1]
        else:
            new_labellist = [1, 0]
        # tf.print(f"pred is {predictions}, label is {new_labellist}")
        bce = tf.keras.metrics.binary_crossentropy(new_labellist, predictions)
        lossvalue = tf.cast(bce, dtype="float32")
        return lossvalue

    transformer = Transformer(num_layers, d_model, num_heads, dff, tfconfig, rate=dropout_rate)
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(tfconfig.init_lr, beta_1=0.99, beta_2=0.9999,
                                         # epsilon=1e-12, amsgrad=False)
                                         epsilon=1e-12, amsgrad=True)
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if use_checkpoint:
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)


    # get #1 protein vecs, #2 rna_vecs, #3 rna padding, #4 protein padding, #5 cross padding
    def get_three_veclists(inp, record_num, tfconfig):  # inp = [prot_name, rna_name, label]

        # get RNA features and padded features
        rna_feat_arr = np.load(f"{tfconfig.rna_feature_file}{inp[1]}.npy.npz", allow_pickle=True)

        # get protein features and padded
        protein_feat_arr = np.load(f"{tfconfig.protein_feature_file}{inp[0]}.npy.npz", allow_pickle=True)
        # make protein padding mask
        pro_padding_mask_list = [np.concatenate([[0] * x.shape[0], [1] * (4000 - x.shape[0])]) for x in [protein_feat_arr["unpadded_array"]]]
        pro_padding_mask_list = tf.cast(pro_padding_mask_list, dtype="float32")
        tfconfig.pro_padding_mask_list = pro_padding_mask_list[0:1, tf.newaxis, tf.newaxis, :]

        # make rna padding mask
        rna_padding_mask_list = [np.concatenate([[0] * x.shape[0], [1] * (4001 - x.shape[0])]) for x in [rna_feat_arr["unpadded_array"]]]
        rna_padding_mask_list = tf.cast(rna_padding_mask_list, dtype="float32")
        tfconfig.rna_padding_mask_list = rna_padding_mask_list[0:1, tf.newaxis, tf.newaxis, :]

        # protein padded features
        tfconfig.pvecs_list_padded = protein_feat_arr["padded_array"]
        pvecs_list_unpadded = protein_feat_arr["unpadded_array"]

        # rna padded features
        rvecs_list_padded = rna_feat_arr["padded_array"]
        rvecs_list_unpadded = rna_feat_arr["unpadded_array"]
        rna_vec_list = rvecs_list_padded
        tfconfig.rna_vec_list = tf.convert_to_tensor(rna_vec_list, dtype=tf.float32)

        # cross mask "pad with 1"
        rna_length = np.array(rvecs_list_unpadded).shape[0]
        protein_length = np.array(pvecs_list_unpadded).shape[0]

        # cross mask
        cross_padding_mask_row = [0] * rna_length + [1] * (4001 - rna_length)
        upper_mask = np.array([cross_padding_mask_row] * protein_length)
        lower_mask = np.array([[1] * 4001] * (4000 - protein_length))
        cross_padding_mask_list = np.concatenate([upper_mask, lower_mask])
        cross_padding_mask_list = tf.cast([cross_padding_mask_list], dtype="float32")
        tfconfig.cross_padding_mask_list = cross_padding_mask_list[tf.newaxis, :, :]

        # protein self mask
        self_pro_mask_row = [0] * protein_length + [1] * (4000 - protein_length)
        upper_mask = np.array([self_pro_mask_row] * protein_length)
        lower_mask = np.array([[1] * 4000] * (4000 - protein_length))
        self_pro_mask_list = np.concatenate([upper_mask, lower_mask])
        self_pro_mask_list = tf.cast([self_pro_mask_list], dtype="float32")
        tfconfig.self_pro_mask_list = self_pro_mask_list[tf.newaxis, :, :]

        # rna self mask
        self_rna_mask_row = [0] * rna_length + [1] * (4001 - rna_length)
        upper_mask = np.array([self_rna_mask_row] * rna_length)
        lower_mask = np.array([[1] * 4001] * (4001 - rna_length))
        self_rna_mask_list = np.concatenate([upper_mask, lower_mask])
        self_rna_mask_list = tf.cast([self_rna_mask_list], dtype="float32")
        tfconfig.self_rna_mask_list = self_rna_mask_list[tf.newaxis, :, :]

        # cross mask small
        # cross_padding_mask_row_small = [0] * rna_length + [1] * (4001 - rna_length)
        # upper_mask = np.array([cross_padding_mask_row_small] * rna_length)
        # lower_mask = np.array([[0] * 4001] * (4001 - rna_length))
        # cross_padding_mask_list_small = np.concatenate([upper_mask, lower_mask])
        # cross_padding_mask_list_small = tf.cast(cross_padding_mask_list_small, dtype="float32")
        # tfconfig.cross_padding_mask_list_small = cross_padding_mask_list_small[tf.newaxis, tf.newaxis, :, :]

        # label list
        label_list = [int(inp[2].strip())]
        tfconfig.label_list = tf.convert_to_tensor(label_list, dtype=tf.int32)

        return tfconfig, label_list


    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    ]

    # @tf.function()
    @tf.function(input_signature=train_step_signature)
    def train_step(inp, m, record_count, tfcfig, losssum):  # tar : (100, 30, 170)
        # get protein seqs, and make masks
        # tar_inp, inp, labellist, pro_padding_mask, rna_padding, cross_mask, cross_mask_small \
        #     = get_three_veclists(inp, record_num, tfconfig)  # protein features vecs
        tfcfig, labellist = get_three_veclists(inp, record_count, tfcfig)
        if tfcfig.pvecs_list_padded is None:
            return [None, m]
        with tf.GradientTape() as tape:
            predictions, _ = transformer.call(True, tfcfig)
            loss2 = loss_function(tfcfig.label_list, predictions)

        gradients = tape.gradient(loss2, transformer.trainable_variables)
        m.update_state(tfcfig.label_list, tf.nn.softmax(predictions)[:, 1].numpy())
        losssum += loss2.numpy()
        # get gradients of this tape
        # Accumulate the gradients
        if record_count % tfcfig.num_accum_grad == 0:
            tfcfig.accum_gradient = gradients
        else:
            tfcfig.accum_gradient = [(acum_grad + grad) if acum_grad is not None else None for acum_grad, grad in zip(tfcfig.accum_gradient, gradients)]
        if record_count % tfcfig.num_accum_grad == 0 and record_count != 0:
            optimizer.apply_gradients(zip(tfcfig.accum_gradient, transformer.trainable_variables))
            tf.print(f"AUROC:{auc.result().numpy()}, Loss: {losssum / tfcfig.num_accum_grad}, Label: {labellst}")
            losssum = 0
            # tfconfig.train_vars = tf.compat.v1.trainable_variables()
            # tfcfig.accum_gradient = [tf.zeros_like(this_var) for this_var in tfcfig.train_vars]
        return loss2, m, labellist, losssum, tfcfig

    # make instances for evaluating optimization
    epoch_loss = tf.keras.metrics.Mean(name='train_loss')
    epoch_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
    auc = tf.keras.metrics.AUC()

    arr = np.load(tfconfig.cv_list_file, allow_pickle=True)
    # load test set id list
    training_list = arr["list"][tfconfig.group_to_ignore]

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f"{BASE_PATH}/logs/{current_time}_start/"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    labellst = None
    for epoch in range(max_epoch):
        # 1 EPOCH
        auc.reset_states()
        epoch_loss.reset_states()
        epoch_accuracy.reset_states()
        record_num, data_count, loss_sum = 0, 0, 0  # confirmed that cv_id_list starts with 0

        # get trainable variables

        # tfconfig.train_vars = transformer.trainable_variables()  # error here variables not created
        # Create empty gradient list (not a tf.Variable list)

        with open(tfconfig.pairfile) as f:
            for lines in f.readlines():
                # check if the record is in the training list
                if record_num not in training_list:
                    ele = lines.split("\t")
                    tfconfig.pair_name = f"{ele[0]}_{ele[1]}"
                    loss, auc, labellst, loss_sum, tfconfig = train_step(ele, auc, data_count, tfconfig, loss_sum)
                    if loss is None:
                        continue
                    # epoch_accuracy.update_state(accuracy)
                    if data_count % 10 == 0:
                        ckpt_manager.save()
                    data_count += 1
                    epoch_loss.update_state(loss)
                record_num += 1
        tf.print(f"EPOCH:{epoch} AUROC:{auc.result().numpy()} LOSS:{epoch_loss.result().numpy()}")
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', epoch_loss.result(), step=epoch)
            tf.summary.scalar('auroc', auc.result(), step=epoch)
        # tf.print(f"EPOCH:{epoch} AUROC:{auc.result().numpy()} ACCURACY:{epoch_accuracy.result().numpy()}")


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    config = Tfconfig()
    config.benchname = sys.argv[3]
    config.init_lr = 10 ** (- int(sys.argv[2]))
    config.num_layers = 2
    config.num_accum_grad = int(sys.argv[5])
    config.num_heads = 4
    config.use_attn_augument = sys.argv[4]
    config.group_to_ignore = int(sys.argv[1])
    config.update()
    opt_trfmr(config)
