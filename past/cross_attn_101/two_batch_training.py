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
import tensorflow_addons as tfa

tf.config.run_functions_eagerly(True)
# np.set_printoptions(threshold=sys.maxsize)
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# gpu_options = tf.GPUOptions(allow_growth=True)

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------


def make_seq_dict(path):
    mask_dict = {}
    with open(path) as f:
        for lines in f.readlines():
            if ">" in lines:
                name = lines[1:].strip()
            else:
                mask_dict[name]=lines.strip()
    return mask_dict


########################################################################
# transformer
# variables to consider
########################################################################
# BASE_PATH = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/"
BASE_PATH = "/gs/hs0/tga-science/kimura/transformer_tape_dnabert/"
PCODES = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
RCODES = ["A", "G", "C", "U"]
WARMUP_STEPS = 5
USE_CHECKPOINT = False
TRAINING = True
MAX_EPOCHS = 10000
D_MODEL = 64
DFF = 64
RNA_MAX_LENGTH = 101
PROTEIN_MAX_LENGTH = 3680
DROPOUT_RATE = 0.1
final_target_vocab_size = 2
########################################################################
########################################################################

path = BASE_PATH


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
        self.warmup_steps = 0

    def update(self, task_id):
        if int(self.use_attn_augument) == 1:  # num to string for task name
            self.aug_or_not = "aug"
        else:
            self.aug_or_not = "noaug"
        self.training_npz_dir = f"{BASE_PATH}data/training_data_tokenized/{self.group_to_ignore}/"
        self.statistical_pot_path = f"{BASE_PATH}data/attn_arrays_hb/{self.group_to_ignore}/"  # 1a1t-A_1a1t-B.npz
        self.statistical_pot_pi_path = f"{BASE_PATH}data/attn_arrays_pi/{self.group_to_ignore}/"  # 1a1t-A_1a1t-B.npz
        self.taskname = f"{task_id}"
        self.checkpoint_path = f"{BASE_PATH}_{self.taskname}"


# ------------------------------------------------
# Functions
# ------------------------------------------------


def get_stat_pot(tfcfg, i):
    arr_hb = np.load(f"{tfcfg.statistical_pot_path}{i}.npy", allow_pickle=True)
    arr_pi = np.load(f"{tfcfg.statistical_pot_pi_path}{i}.npy", allow_pickle=True)
    return arr_hb, arr_pi


def scaled_dot_product_attention(k, q, v, tffig, one_on_rna, w_aug=None, w_at=None):
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
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  # (1, 34, 29, 29)
    if one_on_rna != 1:
        if scaled_attention_logits.shape[-1] != mask.shape[-1]:
            mask = tf.transpose(mask, perm=[0, 1, 3, 2])
        scaled_attention_logits += (mask * -1e9)

    ########################################################
    # add astatistical potentials at Cross Attention Layers
    ########################################################
    if one_on_rna == 2 and tffig.use_attn_augument == 1:
        table_to_augment_hb = tffig.statpot_hb
        table_to_augment_pi = tffig.statpot_pi
        scaled_attention_logits *= w_at
        table_to_augment = tf.stack([table_to_augment_hb, table_to_augment_pi])
        if scaled_attention_logits.shape[-1] != table_to_augment.shape[-1]:
            table_to_augment = tf.transpose(table_to_augment, perm=[0, 2, 1])
        scaled_attention_logits += w_aug * tf.cast(table_to_augment, dtype="float32")
    if tffig.two_d_softm == 1:
        attention_weights = tf.cast(tf.keras.activations.softmax(scaled_attention_logits, axis=[2, 3]), tf.float32)
        attention_weights *= attention_weights.shape[2]
    else:
        attention_weights = tf.cast(tf.keras.activations.softmax(scaled_attention_logits, axis=3), tf.float32)
    output = tf.matmul(tf.transpose(attention_weights, perm=[0, 1, 3, 2]), v)  # (..., seq_len_q, depth_v)
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

    def call(self, x1, x2):
        x = tf.keras.layers.Average()([tf.reduce_mean(x1, axis=1), tf.reduce_mean(x2, axis=1)])
        # x = tf.concat([tf.reduce_mean(x1, axis=1), tf.reduce_mean(x2, axis=1)], axis=1)
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
        pout = self.p_encoders.call(tf_cfg, pout)
        rout = self.r_encoders.call(tf_cfg, rout)
        rout, pout = self.cross_layers.call(tf_cfg, rout, pout)
        prediction = self.final_layer.call(rout, pout)
        return prediction


class Embedders(tf.keras.layers.Layer):  # RNA
    def __init__(self, encfg, one_if_rna):
        super(Embedders, self).__init__()
        if one_if_rna == 1:
            maxlen = encfg.max_rna_len
            vocab_size = 4
        elif one_if_rna == 0:
            maxlen = encfg.max_pro_len
            vocab_size = 20
        self.embedding = tf.keras.layers.Embedding(vocab_size + 1, encfg.d_model)  # 5 because 4 bases and an unknown
        self.d_model = tf.cast(encfg.d_model, tf.float32)
        self.pos_encoding = positional_encoding(maxlen, encfg.d_model)
        self.dropout = tf.keras.layers.Dropout(encfg.dropout_rate)

    def call(self, tf_cfg, one_when_rna):  # tfcfg, RCODES, i, 1, self.rna_out
        # embedding
        if one_when_rna == 1:
            sequence = tf_cfg.rnatok
            maxlen = tf_cfg.max_rna_len
        elif one_when_rna == 0:
            sequence = tf_cfg.protok
            maxlen = tf_cfg.max_pro_len
        x = self.embedding(sequence)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # x=(1, 662, 64)
        # padding
        if one_when_rna == 0:
            zero_pad = tf.repeat([[0] * tf_cfg.d_model], repeats=(maxlen - x.shape[1]), axis=0)
            zero_pad = tf.cast(zero_pad, tf.float32)
            x = tf.concat([x[0, :, :], zero_pad], axis=0)  # x=(3680, 64)
            x += self.pos_encoding[:len(tf_cfg.protok), :]
        else:
            x += self.pos_encoding[:len(tf_cfg.rnatok), :]
        # dropout
        x = self.dropout(x, training=tf_cfg.training)
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
        for i in range(self.layer_num):
            self.rencoder = SelfAttention(trscfg, one_if_rna)

    def call(self, tfcfg, x):
        for i in range(self.layer_num):
            x = self.rencoder.call(tfcfg, x)
        return x


class CrossLayers(tf.keras.layers.Layer):
    def __init__(self, trscfg):
        super(CrossLayers, self).__init__()
        for i in range(trscfg.cross_layer_num):
            self.cross_attention = CrossAttention(trscfg)

    def call(self, tfcfg, rna_out, pro_out):
        for i in range(tfcfg.cross_layer_num):
            pro_out, rna_out = self.cross_attention.call(tfcfg, pro_out, rna_out)  # q, kv, confg, layr_num, one_when_rna
        return pro_out, rna_out


class CrossAttention(tf.keras.layers.Layer):  # two inputs/outputs
    def __init__(self, cr_cfg):
        super(CrossAttention, self).__init__()
        self.rna_cross_attention = AttentionLayer(cr_cfg, cr_cfg.cross_dff)
        self.pro_cross_attention = AttentionLayer(cr_cfg, cr_cfg.cross_dff)

    def call(self, cross_cfg, pro_inp, rna_inp):
        # q, kv, confg, layr_num, one_when_rna
        rna_out = self.rna_cross_attention.call(rna_inp, pro_inp, cross_cfg, 2)
        pro_out = self.pro_cross_attention.call(pro_inp, rna_inp, cross_cfg, 2)
        return pro_out, rna_out


class SelfAttention(tf.keras.layers.Layer):  # one input/output
    def __init__(self, encfg, one_if_rna):
        super(SelfAttention, self).__init__()
        self.one_if_rna = one_if_rna
        if one_if_rna == 1:
            dff = encfg.rna_dff
        else:
            dff = encfg.pro_dff
        self.self_layer = AttentionLayer(encfg, dff)
        self.dropout = tf.keras.layers.Dropout(encfg.dropout_rate)

    def call(self, tf_cfg, x=None):  # tfcfg, RCODES, i, 1, self.rna_out
        x = self.self_layer.call(x, x, tf_cfg, self.one_if_rna)
        return x


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
        attn_output = self.dropout1(attn_output, training=confg.training)
        out1 = self.layernorm1(q + attn_output)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=confg.training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, mhacfg):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = mhacfg.num_heads
        self.d_model = mhacfg.d_model
        assert mhacfg.d_model % self.num_heads == 0
        self.depth = mhacfg.d_model // self.num_heads  # depth = 128 /4 = 32
        self.wq = tf.keras.layers.Dense(mhacfg.d_model)
        self.wk = tf.keras.layers.Dense(mhacfg.d_model)
        self.wv = tf.keras.layers.Dense(mhacfg.d_model)
        self.dense = tf.keras.layers.Dense(mhacfg.d_model)
        if mhacfg.use_attn_augument == 1:
            self.aug_weight_aug = tf.compat.v1.get_variable(
                "aug_weight_aug", [1],
                trainable=True,
                dtype="float32",
                initializer=tf.random_uniform_initializer(maxval=100, minval=0))
            self.aug_weight_at = tf.compat.v1.get_variable(
                "aug_weight_at", [1],
                trainable=True,
                dtype="float32",
                initializer=tf.random_uniform_initializer(maxval=100, minval=0))
                # constraint=lambda t: tf.clip_by_value(t, 0, 1))


    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, kv, tfcg, one_for_rna):
        batch_size = tf.shape(q)[0]
        v = self.wv(kv)
        k = self.wk(kv)
        q = self.wq(q)
        # split
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        # calc. attention
        if tfcg.use_attn_augument == 1:
            scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, tfcg, one_for_rna,
                                                                             self.aug_weight_aug, self.aug_weight_at)
        else:
            scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, tfcg, one_for_rna)
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


def get_three_veclists(tffg):
    # 1. get protein sequence, change to a 128 length feature vec

    # 2. make masks
    # protein self mask
    protein_length = len(tffg.protok[0])
    self_pro_mask_row = [0] * protein_length + [1] * (tffg.max_pro_len - protein_length)
    upper_mask = tf.repeat([self_pro_mask_row], repeats=protein_length, axis=0)
    # lower_mask = np.array([[1] * tffg.max_pro_len] * (tffg.max_pro_len - protein_length))
    lower_mask = tf.constant(1, shape=(tffg.max_pro_len - protein_length, tffg.max_pro_len))
    # self_pro_mask_list = np.concatenate([upper_mask, lower_mask])
    self_pro_mask_list = tf.concat([upper_mask, lower_mask], axis=0)
    self_pro_mask_list = tf.cast([self_pro_mask_list], dtype="float32")
    tffg.self_pro_mask_list = self_pro_mask_list[tf.newaxis, :, :]

    # rna self mask
    rna_length = len(tffg.rnatok)

    # cross mask
    cross_padding_mask_row = [0] * rna_length + [1] * (tffg.max_rna_len - rna_length)
    upper_mask = tf.repeat([cross_padding_mask_row], repeats=protein_length, axis=0)
    lower_mask = tf.repeat([[1] * tffg.max_rna_len], repeats=(tffg.max_pro_len - protein_length), axis=0)
    cross_padding_mask_list = tf.concat([upper_mask, lower_mask], 0)
    cross_padding_mask_list = tf.cast([cross_padding_mask_list], dtype="float32")
    tffg.cross_padding_mask_list = cross_padding_mask_list[tf.newaxis, :, :]

    # 3. make padded features max RNA 4000、pro 3680
    # Do this in a layer

    # 4. make a label list
    if tffg.label == [1]:
        tffg.new_label_list = tf.convert_to_tensor([[0, 1]], dtype="float32")
    else:
        tffg.new_label_list = tf.convert_to_tensor([[1, 0]], dtype="float32")

    return tffg


def opt_trfmr(tfconfig):
    # ------------------------------------------------
    # Constant
    # ------------------------------------------------

    max_epoch = tfconfig.max_epoch
    checkpoint_path = tfconfig.checkpoint_path
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)


    def loss_function(labellist, predictions):
        bce = tf.keras.metrics.binary_crossentropy(labellist, predictions)
        lossvalue = tf.cast(bce, dtype="float32")
        return lossvalue


    transformer = Transformer(tfconfig)
    # learning_rate = CustomSchedule(tfconfig.d_model, tfconfig.warmup_steps)
    learning_rate = tfconfig.init_lr
    # optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9, amsgrad=True)
    optimizer = tfa.optimizers.RectifiedAdam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9, weight_decay=1e-5)
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if tfconfig.usechpoint == 1:
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    ]

    # @tf.function()
    @tf.function(input_signature=train_step_signature)
    def train_step(m, record_count, tfcfig, losssum):
        tfcfig = get_three_veclists(tfcfig)

        with tf.GradientTape() as tape:
            predictions = transformer(tfcfig)
            loss2 = loss_function(tfcfig.new_label_list, predictions)
        gradients = tape.gradient(loss2, transformer.trainable_variables)
        m.update_state(tfcfig.new_label_list, tf.nn.softmax(predictions).numpy())
        losssum += loss2.numpy()
        # ###################################
        # Accumulate gradients
        # ###################################
        if tfcfig.accum_gradient is None:  # first record
            tfcfig.accum_gradient = [flat_gradients(grad) for grad in gradients if grad is not None]
            # tfcfig.accum_gradient = gradients
        elif record_count % tfcfig.num_accum_grad == 0:  # apply grads to change vars
            trainable_variables = [var for grad, var in zip(gradients, transformer.trainable_variables) if grad is not None]
            gradients = [flat_gradients(grad) for grad, var in zip(gradients, transformer.trainable_variables) if grad is not None]
            optimizer.apply_gradients(zip(tfcfig.accum_gradient, trainable_variables))
            losssum = 0
            tfcfig.accum_gradient = [flat_gradients(grad) for grad in gradients if grad is not None]
        else:  # continue adding grads
            gradients = [flat_gradients(grad) for grad in gradients if grad is not None]
            tfcfig.accum_gradient = [(acum_grad + grad) for acum_grad, grad in zip(tfcfig.accum_gradient, gradients)]
        return loss2, m, losssum, tfcfig

    # make instances for evaluating optimization
    epoch_loss = tf.keras.metrics.Mean(name='train_loss')
    epoch_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
    auc = tf.keras.metrics.AUC()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f"{BASE_PATH}/logs/{tfconfig.taskname}_{current_time}/"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    for epoch in range(max_epoch):
        auc.reset_states()
        epoch_loss.reset_states()
        epoch_accuracy.reset_states()
        record_num, data_count, loss_sum, num_in_batch = 0, 0, 0, 0  # confirmed that cv_id_list starts with 0

        for i in range(tfconfig.max_file_num):
            arr = np.load(f"{tfconfig.training_npz_dir}{i}.npz", allow_pickle=True)
            pot_arr_hb, ppot_arr_pi = get_stat_pot(tfconfig, i)
            for proid, protok, rnatok, label, pot_hb_table, pot_pi_table in zip(arr["proid"], arr["protok"], arr["rnatok"], arr["label"], pot_arr_hb, ppot_arr_pi):
                tfconfig.proid = tf.convert_to_tensor(proid, dtype="int16")
                tfconfig.protok = tf.convert_to_tensor(protok, dtype="float32")
                tfconfig.protok = tfconfig.protok[tf.newaxis, :]
                tfconfig.rnatok = tf.convert_to_tensor(rnatok, dtype="float32")
                tfconfig.rnatok = tfconfig.rnatok[tf.newaxis, :]
                tfconfig.label = tf.convert_to_tensor(label, dtype="int16")
                tfconfig.statpot_hb = tf.convert_to_tensor(pot_hb_table, dtype="float32")
                tfconfig.statpot_pi = tf.convert_to_tensor(pot_pi_table, dtype="float32")

                loss, auc, loss_sum, tfconfig = train_step(auc, data_count, tfconfig, loss_sum)
                if loss is None:
                    continue
                data_count += 1
                epoch_loss.update_state(loss)
            # tf.print(f"AUROC:{auc.result().numpy()} LOSS:{epoch_loss.result().numpy()} LR:{optimizer.lr.numpy()}")
        ckpt_manager.save()
        tf.print(f"EPOCH:{epoch} AUROC:{auc.result().numpy()} LOSS:{epoch_loss.result().numpy()}")
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', epoch_loss.result(), step=epoch)
            tf.summary.scalar('auroc', auc.result(), step=epoch)


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
    config.training = TRAINING
    config.group_to_ignore = int(sys.argv[1])
    config.warmup_steps = int(sys.argv[2])
    config.use_attn_augument = int(sys.argv[3])
    config.num_accum_grad = int(sys.argv[4])
    config.two_d_softm = int(sys.argv[5])
    config.num_heads = int(sys.argv[6])
    config.max_file_num = int(sys.argv[7])
    config.usechpoint = int(sys.argv[8])
    config.init_lr = 10 ** (-int(sys.argv[9]))
    config.self_player_num = int(sys.argv[10])
    config.self_rlayer_num = int(sys.argv[11])
    config.cross_layer_num = int(sys.argv[12])
    config.rna_dff = int(sys.argv[13]) * 64
    config.pro_dff = int(sys.argv[14]) * 64
    config.cross_dff = int(sys.argv[15]) * 64
    config.batch_size = int(sys.argv[16])
    task_identifier = f"{sys.argv[1:]}"
    task_identifier = task_identifier.replace(", ", "_").replace("[", "").replace("]", "").replace("'", "")
    config.update(task_identifier)
    opt_trfmr(config)
