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
RNA_MAX_LENGTH = 4000
PROTEIN_MAX_LENGTH = 3680
DROPOUT_RATE = 0.1
final_target_vocab_size = 2
########################################################################
########################################################################

path = BASE_PATH


class Tfconfig():
    def __init__(self):
        self.max_pro_len=PROTEIN_MAX_LENGTH
        self.max_rna_len=RNA_MAX_LENGTH
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


    def update(self):
        if int(self.use_attn_augument) == 1:  # num to string for task name
            self.aug_or_not = "aug"
        else:
            self.aug_or_not = "noaug"
        self.pairfile = f"{BASE_PATH}data/benchmarks/label/{self.benchname}_pairs_shuffled.txt"
        self.rna_seq_file = f"{BASE_PATH}data/benchmarks/sequence/{self.benchname}_rna_seq.fa"  # AARS.npy
        self.protein_seq_file = f"{BASE_PATH}data/benchmarks/sequence/{self.benchname}_protein_seq.fa"  # 0.npy
        self.pro_seq_dict = make_seq_dict(self.protein_seq_file)
        self.rna_seq_dict = make_seq_dict(self.rna_seq_file)
        # "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/finetune_bench/RPI369"
        self.cv_list_file = f"{BASE_PATH}data/benchmarks/id_list_{self.benchname}_for_5CV.csv.npz"
        self.statistical_pot_path = f"{BASE_PATH}data/attn_arrays_no_tape_no_dnabert/{self.benchname}/"  # 1a1t-A_1a1t-B.npz
        self.statistical_pot_pi_path = f"{BASE_PATH}data/attn_arrays_pi_no_tape_no_dnabert/{self.benchname}/"  # 1a1t-A_1a1t-B.npz
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


def scaled_dot_product_attention(k, q, v, tffig, lynum, one_on_rna,
                                 augweight_hb=None, augweight_pi=None, augweight_at=None):
    # one_on_rna = 1:rna-self, 0:protein-self, 2:cross

    # calculate attention table
    matmul_qk = tf.cast(tf.matmul(q, k, transpose_b=True), tf.float32)  # (..., seq_len_q, seq_len_k)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    # assign smask in Decoder's 1st layer

    if one_on_rna == 1:  # rna self
        mask = tffig.self_rna_mask_list
    elif one_on_rna == 0:  # pro self
        mask = tffig.self_pro_mask_list
    else:            # cross
        mask = tffig.cross_padding_mask_list

    v = tf.cast(v, tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  # (1, 34, 29, 29)
    if scaled_attention_logits.shape[-1] != mask.shape[-1]:
        mask = tf.transpose(mask, perm=[0, 1, 3, 2])
    scaled_attention_logits += (mask * -1e9)

    ########################################################
    # add astatistical potentials at Cross Attention Layers
    ########################################################
    if one_on_rna == 2 and tffig.use_attn_augument == 1:
        table_to_augment = get_stat_pot_arr(tffig)
        table_to_augment_pi = get_pi_stat_pot_arr(tffig)
        scaled_attention_logits *= augweight_at[lynum]
        scaled_attention_logits += augweight_hb[lynum] * table_to_augment
        scaled_attention_logits += augweight_pi[lynum] * table_to_augment_pi
    #     attention_weights = tf.cast(tf.keras.activations.softmax(scaled_attention_logits, axis=[2, 3]), tf.float32)
    # else:
    #     attention_weights = tf.keras.activations.softmax(scaled_attention_logits, axis=[2, 3])
    if tffig.two_d_softm == 1:
        attention_weights = tf.cast(tf.keras.activations.softmax(scaled_attention_logits, axis=[2, 3]), tf.float32)
        attention_weights *= attention_weights.shape[2]
    else:
        attention_weights = tf.cast(tf.keras.activations.softmax(scaled_attention_logits, axis=3), tf.float32)
    # output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
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
        self.attn_layers = RepeatLayers(trcfg)
        self.final_layer = FinalLayer()

    def call(self, tf_cfg):
        rout, pout = self.attn_layers.call(tf_cfg)
        prediction = self.final_layer.call(rout, pout)
        return prediction


class RepeatLayers(tf.keras.layers.Layer):
    def __init__(self, trscfg):
        super(RepeatLayers, self).__init__()
        for i in range(trscfg.layer_num_self):
            self.rencoder = EmbAttention(trscfg, RCODES, 1)
            self.pencoder = EmbAttention(trscfg, PCODES, 0)
        for i in range(trscfg.layer_num_cross):
            self.cross_attention = CrossAttention(trscfg)

    def call(self, tfcfg):
        rna_out, pro_out = None, None
        for i in range(tfcfg.layer_num_self):
            rna_out = self.rencoder.call(tfcfg, RCODES, i, 1, rna_out)
            pro_out = self.pencoder.call(tfcfg, PCODES, i, 0, pro_out)
        for i in range(tfcfg.layer_num_cross):
            pro_out, rna_out = self.cross_attention.call(tfcfg, i, pro_out, rna_out)  # q, kv, confg, layr_num, one_when_rna
        return pro_out, rna_out


class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, cr_cfg):
        super(CrossAttention, self).__init__()
        self.rna_cross_attention = AttentionLayer(cr_cfg)
        self.pro_cross_attention = AttentionLayer(cr_cfg)

    def call(self, cross_cfg, laynum, pro_inp, rna_inp):
        # q, kv, confg, layr_num, one_when_rna
        rna_out = self.rna_cross_attention.call(rna_inp, pro_inp, cross_cfg, laynum, 2)
        pro_out = self.pro_cross_attention.call(pro_inp, rna_inp, cross_cfg, laynum, 2)
        return pro_out, rna_out


class EmbAttention(tf.keras.layers.Layer):  # RNA
    def __init__(self, encfg, vocab_list, one_if_rna):
        super(EmbAttention, self).__init__()
        self.one_if_rna = one_if_rna
        if self.one_if_rna == 1:
            self.sequence = encfg.rnaseq
            self.maxlen = encfg.max_rna_len
        elif self.one_if_rna == 0:
            self.sequence = encfg.proseq
            self.maxlen = encfg.max_pro_len
        self.embedding = tf.keras.layers.Embedding(len(vocab_list) + 1, encfg.d_model)  # 5 because 4 bases and an unknown
        self.d_model = tf.cast(encfg.d_model, tf.float32)
        self.pos_encoding = positional_encoding(self.maxlen, encfg.d_model)
        if encfg.use_self == 1:
            self.self_layer = AttentionLayer(encfg)
        self.dropout = tf.keras.layers.Dropout(encfg.dropout_rate)

    def call(self, tf_cfg, vocablist, layernum, one_when_rna, x=None):
        if one_when_rna == 1:
            self.sequence = tf_cfg.rnaseq
            self.maxlen = tf_cfg.max_rna_len
        elif one_when_rna == 0:
            self.sequence = tf_cfg.proseq
            self.maxlen = tf_cfg.max_pro_len
        if layernum == 0:  # Embedding at layer #0
            x = self.embedding(np.array([vocablist.index(x) if x in vocablist else len(vocablist) for x in self.sequence]))
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            zero_pad = np.array([[0] * tf_cfg.d_model] * (self.maxlen - x.shape[0]))
            x = tf.concat([x, zero_pad], axis=0)  # x=[99,64], zero_pad=[3901,1]
            x += self.pos_encoding[:len(self.sequence), :]
            x = self.dropout(x, training=tf_cfg.training)
            x = tf.cast(x, tf.float32)
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            x /= tf.cast(self.d_model, tf.float32)
        else:
            pass
        if tf_cfg.use_self == 1:
            x = self.self_layer.call(x, x, tf_cfg, layernum, self.one_if_rna)
        else:
            pass
        return x


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, elcfg):
        super(AttentionLayer, self).__init__()
        self.mha = MultiHeadAttention(elcfg)
        self.ffn = point_wise_feed_forward_network(elcfg.d_model, elcfg.dff)
        if elcfg.use_layernorm == 1:
            self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-9)
            self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-9)
        self.dropout1 = tf.keras.layers.Dropout(elcfg.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(elcfg.dropout_rate)

    def call(self, q, kv, confg, layr_num, one_when_rna):
        attn_output, _ = self.mha.call(q, kv, confg, layr_num, one_when_rna)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=confg.training)
        if confg.use_layernorm == 1:
            out1 = self.layernorm1(q + attn_output)  # (batch_size, input_seq_len, d_model)
        else:
            out1 = q + attn_output
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=confg.training)
        if confg.use_layernorm == 1:
            out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        else:
            out2 = out1 + ffn_output
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
            self.aug_weight_hb = tf.compat.v1.get_variable(
                "aug_weight_hb",
                shape=[mhacfg.num_layers_cross, 1],
                trainable=True,
                dtype="float32",
                initializer=tf.random_uniform_initializer(maxval=1, minval=0),
                constraint=lambda t: tf.clip_by_value(t, 0, 1))
            self.aug_weight_pi = tf.compat.v1.get_variable(
                "aug_weight_pi",
                shape=[mhacfg.num_layers_cross, 1],
                trainable=True,
                dtype="float32",
                initializer=tf.random_uniform_initializer(maxval=1, minval=0),
                constraint=lambda t: tf.clip_by_value(t, 0, 1))
            self.aug_weight_at = tf.compat.v1.get_variable(
                "aug_weight_at",
                shape=[mhacfg.num_layers_cross, 1],
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

    def call(self, q, kv, tfcg, lay_num, one_for_rna):
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
            scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, tfcg, lay_num, one_for_rna,
                                                            self.aug_weight_hb, self.aug_weight_pi, self.aug_weight_at)
        else:
            scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, tfcg, lay_num, one_for_rna)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights


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


def get_three_veclists(inp, record_num, tfconfig3):  # inp = [prot_name, rna_name, label]
    # 1. get protein sequence, change to a 128 length feature vec
    tfconfig = tfconfig3
    tfconfig.proseq = tfconfig.pro_seq_dict[inp[0]]
    tfconfig.rnaseq = tfconfig.rna_seq_dict[inp[1]]

    # 2. make masks
    # protein self mask
    protein_length = len(tfconfig.proseq)
    self_pro_mask_row = [0] * protein_length + [1] * (tfconfig.max_pro_len - protein_length)
    upper_mask = np.array([self_pro_mask_row] * protein_length)
    lower_mask = np.array([[1] * tfconfig.max_pro_len] * (tfconfig.max_pro_len - protein_length))
    self_pro_mask_list = np.concatenate([upper_mask, lower_mask])
    self_pro_mask_list = tf.cast([self_pro_mask_list], dtype="float32")
    tfconfig.self_pro_mask_list = self_pro_mask_list[tf.newaxis, :, :]

    # rna self mask
    rna_length = len(tfconfig.rnaseq)
    self_rna_mask_row = [0] * rna_length + [1] * (tfconfig.max_rna_len - rna_length)
    upper_mask = np.array([self_rna_mask_row] * rna_length)
    lower_mask = np.array([[1] * tfconfig.max_rna_len] * (tfconfig.max_rna_len - rna_length))
    self_rna_mask_list = np.concatenate([upper_mask, lower_mask])
    self_rna_mask_list = tf.cast([self_rna_mask_list], dtype="float32")
    tfconfig.self_rna_mask_list = self_rna_mask_list[tf.newaxis, :, :]

    # cross mask
    cross_padding_mask_row = [0] * rna_length + [1] * (tfconfig.max_rna_len - rna_length)
    upper_mask = np.array([cross_padding_mask_row] * protein_length)
    lower_mask = np.array([[1] * tfconfig.max_rna_len] * (tfconfig.max_pro_len - protein_length))
    cross_padding_mask_list = np.concatenate([upper_mask, lower_mask])
    cross_padding_mask_list = tf.cast([cross_padding_mask_list], dtype="float32")
    tfconfig.cross_padding_mask_list = cross_padding_mask_list[tf.newaxis, :, :]

    # 3. make padded features max RNA 4000、pro 3680
    # Do this in a layer

    # 4. make a label list
    label_list = [int(inp[2].strip())]
    if label_list[0] == 1:
        tfconfig.new_labellist = tf.convert_to_tensor([[0, 1]])
    else:
        tfconfig.new_labellist = tf.convert_to_tensor([[1, 0]])
    tfconfig.label_list = tf.convert_to_tensor(label_list, dtype=tf.int32)

    return tfconfig, label_list


def opt_trfmr(tfconfig):

    # ------------------------------------------------
    # Constant
    # ------------------------------------------------
    use_checkpoint = USE_CHECKPOINT

    max_epoch = tfconfig.max_epoch
    checkpoint_path = tfconfig.checkpoint_path
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)


    def loss_function(labellist, predictions):
        bce = tf.keras.metrics.BinaryCrossentropy()
        loss = bce(y_true=labellist, y_pred=tf.convert_to_tensor(predictions[0]))
        lossvalue = tf.cast(loss, dtype="float32")
        return lossvalue


    transformer = Transformer(tfconfig)
    optimizer = tf.keras.optimizers.Adam(tfconfig.init_lr, beta_1=0.9, beta_2=0.99,
                                         epsilon=1e-5, amsgrad=True)
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if use_checkpoint:
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    ]

    # @tf.function()
    @tf.function(input_signature=train_step_signature)
    def train_step(inp, m, record_count, tfcfig, losssum):
        tfcfig, labellist = get_three_veclists(inp, record_count, tfcfig)

        with tf.GradientTape() as tape:
            predictions = transformer.call(tfcfig)
            loss2 = loss_function(tfcfig.new_labellist, predictions)
        gradients = tape.gradient(loss2, transformer.trainable_variables)
        tf.print(f"gradients {gradients}")
        m.update_state(tfcfig.new_labellist, predictions.numpy())
        losssum += loss2.numpy()
        # ###################################
        # Accumulate gradients
        # ###################################
        tf.print(f"gradients before {gradients}")  # (None, None, ...)????
        gradients = [flat_gradients(x) for x in gradients]

        if tfcfig.accum_gradient is None:  # first record
            tfcfig.accum_gradient = gradients
        elif record_count % tfcfig.num_accum_grad == 0:  # apply grads to change vars
            optimizer.apply_gradients(zip(tfcfig.accum_gradient, transformer.trainable_variables))
            tf.print(f"AUROC:{auc.result().numpy()}, Loss: {losssum / tfcfig.num_accum_grad}, Label: {labellst}")
            losssum = 0
            tfcfig.accum_gradient = gradients  # reset accum
        else:  # continue adding grads
            tf.print("case 3")
            tfcfig.accum_gradient = [(acum_grad + grad) for acum_grad, grad in zip(tfcfig.accum_gradient, gradients)]
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


def flat_gradients(grads_or_idx_slices: tf.Tensor) -> tf.Tensor:
    if type(grads_or_idx_slices) == tf.IndexedSlices:
        return tf.scatter_nd(
            tf.expand_dims(grads_or_idx_slices.indices, 1),
            grads_or_idx_slices.values,
            tf.cast(grads_or_idx_slices.dense_shape, dtype="int64")
        )
    return grads_or_idx_slices


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    config = Tfconfig()
    config.benchname = sys.argv[3]
    config.training = TRAINING
    config.init_lr = 10 ** (- int(sys.argv[2]))
    config.layer_num_self = int(sys.argv[8])
    config.layer_num_cross = int(sys.argv[9])
    config.two_d_softm = int(sys.argv[10])
    config.num_accum_grad = int(sys.argv[5])
    config.use_self = int(sys.argv[6])
    config.use_layernorm = int(sys.argv[7])
    config.num_heads = 4
    config.use_attn_augument = sys.argv[4]
    config.group_to_ignore = int(sys.argv[1])
    config.update()
    opt_trfmr(config)
