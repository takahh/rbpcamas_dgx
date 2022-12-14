# -------------------------------------------------------------------
# this is for only predcition --- NO TRAINING
# -------------------------------------------------------------------

# ------------------------------------------------
# Import
# ------------------------------------------------
import numpy as np
import optuna
import tensorflow
import tensorflow as tf
from numpy.random import multivariate_normal as mn
from cmaes import SepCMA
import sys
import os

tf.config.run_functions_eagerly(True)
# tf.enable_eager_execution()
# np.set_printoptions(threshold=sys.maxsize)
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# gpu_options = tf.GPUOptions(allow_growth=True)

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------


def make_protein_seq_dic(path):
    pro_dict = {}
    for files in os.listdir(path):
        if "npy" in files:
            protname = files.split(".")[0]
            arr = np.load(f"{path}{files}", allow_pickle=True)
            pro_dict[protname] = arr
    return pro_dict


########################################################################
# transformer
# variables to consider
########################################################################
BASE_PATH = "/gs/hs0/tga-science/kimura/transformer_tape_dnabert/"
PROTEIN_SEQ_FILE = f"{BASE_PATH}data/protein/"  # AARS.npy
protein_seq_dict = make_protein_seq_dic(PROTEIN_SEQ_FILE)
RNA_SEQ_FILE = f"{BASE_PATH}data/finetune_data/"  # 0.npy
WARMUP_STEPS = 100000
USE_CHECKPOINT = True
MAX_BATCH_NUM = 1
MAX_EPOCHS = 10000
# MAX_BATCH_NUM = 10000
NUM_LAYERS = 4
D_MODEL = 768
NUM_HEADS = 6
DFF = 3000
DROPOUT_RATE = 0.1
PROTNAMES = ['EIF3D', 'HNRNPL', 'BCLAF1', 'CPEB4', 'NSUN2', 'SMNDC1', 'AKAP8L', 'BCCIP', 'EIF3G', 'DDX24', 'METAP2', 'EIF4G2', 'LSM11', 'UCHL5', 'UTP18', 'AKAP1', 'SND1', 'DDX42', 'IGF2BP3', 'PTBP1', 'RBFOX2', 'GPKOW', 'SRSF7', 'STAU2', 'TAF15', 'SAFB2', 'ZC3H8', 'SF3B4', 'HNRNPA1', 'XRN2', 'SUGP2', 'XPO5', 'LIN28B', 'NONO', 'SDAD1', 'ZNF622', 'SFPQ', 'SF3B1', 'UTP3', 'CPSF6', 'LARP7', 'NIPBL', 'ZRANB2', 'KHSRP', 'WRN', 'HNRNPC', 'SRSF1', 'EIF3H', 'ILF3', 'PPIG', 'CSTF2T', 'LARP4', 'SSB', 'AGGF1', 'U2AF2', 'SF3A3', 'YWHAG', 'FMR1', 'U2AF1', 'NCBP2', 'DKC1', 'TRA2A', 'SERBP1', 'ZC3H11A', 'FXR2', 'FXR1', 'SUPV3L1', 'FKBP4', 'RPS11', 'GRSF1', 'ZNF800', 'SUB1', 'PPIL4', 'FUBP3', 'WDR3', 'NOLC1', 'UPF1', 'HNRNPU', 'HNRNPUL1', 'DROSHA', 'DDX52', 'QKI', 'TROVE2', 'DDX51', 'ABCF1', 'TBRG4', 'DDX21', 'XRCC6', 'GTF2F1', 'DGCR8', 'CDC40', 'MATR3', 'PHF6', 'POLR2G', 'NIP7', 'WDR43', 'DHX30', 'FUS', 'MTPAP', 'NPM1', 'HNRNPK', 'SRSF9', 'PABPC4', 'PRPF4', 'DDX3X', 'RPS3', 'AATF', 'SBDS', 'DDX59', 'RBM15', 'DDX6', 'PUM2', 'EXOSC5', 'HLTF', 'PUM1', 'PUS1', 'GEMIN5', 'GRWD1', 'RBM5', 'CSTF2', 'FTO', 'SLTM', 'GNL3', 'EWSR1', 'AARS', 'NKRF', 'FAM120A', 'PCBP2', 'FASTKD2', 'EFTUD2', 'PRPF8', 'RBM22', 'IGF2BP2', 'BUD13', 'SAFB', 'TIAL1', 'NOL12', 'IGF2BP1', 'PCBP1', 'SLBP', 'AQR', 'DDX55', 'APOBEC3C', 'G3BP1', 'KHDRBS1', 'PABPN1', 'TARDBP', 'HNRNPM', 'TIA1', 'YBX3', 'RPS5', 'TNRC6A']
final_target_vocab_size = 2
########################################################################
########################################################################

# BASE_PATH = "/gs/hs0/tga-science/kimura/BBO6_sepCMA/"
# quadrutic functin
SIGMA = 1

# TRAINING
TOTAL_SEEDS_IN_CMA_POP = np.arange(1, 2)  # seed for sampling
SEEDF_MAX = np.arange(0, 1)  # determines the shape of the target function
MAX_GENERATION = 99000

# how to make a training data
SKIP = 0
MEAN_SKIP = 1
LINE_TO_STOP_READ_DATA = None

# how to train
RUN_TRANINIG = True
# RUN_TRANINIG = True
EPOCH_TRAIN = 100
FILE_NUM_TRAIN = 10  # constant, dont change this! defined quadrutic function for data generation
BATCH_NUM = 100  # not using this any more
input_vocab_size = 3000  # looks important but no
target_vocab_size = 3000  # looks important but no

# predict
TARGET_FUNCTION_SEED = 1
PRED_MAX = 5000
PRED_FROM_EPOCH = 1  # From when the prediction starts, from scratch? continue?
SOS = tf.constant([[10] * 768], dtype="float32")
EOS = tf.constant([[11] * 768], dtype="float32")
path = BASE_PATH


class Tfconfig():
    def __init__(self):
        self.num_layers = NUM_LAYERS
        self.d_model = D_MODEL
        self.num_heads = NUM_HEADS
        self.dff = DFF
        self.dropout_rate = DROPOUT_RATE
        self.file_num_to_train = FILE_NUM_TRAIN
        self.group_to_ignore = None
        self.max_epoch = MAX_EPOCHS
        self.max_files_trained = 10
        self.max_files_test = 10
        self.do_attention = False

    def update(self):  # _protein_group_except_0_30files
        self.taskname = f"protein_group_except_{self.group_to_ignore}_{self.max_files_trained}files"
        self.checkpoint_path = f"{BASE_PATH}_{self.taskname}"
        self.testlist = PROTNAMES[self.group_to_ignore * 38:(self.group_to_ignore + 1) * 38]
        # self.protlist = [x for x in PROTNAMES if x not in self.ignorelist]
        self.auc_datapath = f"{BASE_PATH}auc/{self.taskname}_test{self.max_files_test}"


# ------------------------------------------------
# Functions
# ------------------------------------------------


def get_protein_seq(protname):
    arr = np.load(f"{path}{protname}.npy")
    return arr


def is_pos_def(A):
    M = np.matrix(A)
    return np.all(np.linalg.eigvals(M + M.transpose()) > 0)


def get_list_fromlog(keyword, logfile):
    targetlist = []
    with open(logfile) as f:
        for lines in f.readlines():
            if keyword == lines[:len(keyword)]:
                str_list = lines.replace(keyword, "").replace("\n", "").split(",")
                list_to_add = [float(x) for x in str_list]
                targetlist.append(list_to_add)
    return targetlist


def write_all(mean_list, logpath, iternum, q_and_r_list, cov_mat):
    with open(logpath.replace(".csv", "_only_iternum.csv"), "a") as f:
        f.writelines(str(iternum) + "\n")

    with open(logpath, "a") as f:
        # write query and response
        f.writelines(f"{iternum}:q_and_r:")
        f.write(",".join(str(x) for x in q_and_r_list))
        f.writelines("\n")

        # write mean
        f.writelines(f"{iternum}:mean:")
        f.write(",".join(str(x) for x in mean_list))
        f.writelines("\n")

        # write matrix
        # row_count = 0
        f.writelines(f"{iternum}:cov:")
        f.write(",".join(str(x) for x in cov_mat))
        f.writelines("\n")


def iter_num_not_yet(num, logfile):
    with open(logfile.replace(".csv", "_only_iternum.csv")) as f:
        # line_list = f.readlines()
        try:
            lastline = f.readlines()[-1]
            return lastline == str(num) + "\n"
        except IndexError:
            print(num)


def reset_log(file_path):
    with open(file_path, "w"):
        pass
    with open(file_path.replace(".csv", "_only_iternum.csv"), "w"):
        pass


sos = SOS
eos = EOS


def create_tar_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def scaled_dot_product_attention(q, k, v, mask, in_decoder=None):  # q, k, v
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  # (1, 34, 29, 29)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class FinalLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(FinalLayer, self).__init__()
        self.finallayer = tf.keras.layers.Dense(final_target_vocab_size)

    def call(self, x):  # x is a 2d vector
        # normalize
        logits = tf.math.l2_normalize(self.finallayer(x))  # [[-0.953382075 0.301765621]]
        # softmax
        finalvalue = tf.nn.softmax(logits[0])
        return finalvalue


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)
        self.final_layer = FinalLayer()

    def call(self, inp, tar, training, enc_padding_mask,
            dec_padding_mask, cross_padding_mask, wordlen=None):
        enc_output = self.encoder(inp, training, enc_padding_mask, wordlen)  # (batch_size, inp_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, dec_padding_mask, cross_padding_mask, wordlen)
        final_output = self.final_layer(tf.reduce_sum(dec_output, axis=-2))  # output: 1-dimensional values
        return final_output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wk2 = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask, training, wordlen=None, block_num=None, mha_num=None):
        batch_size = tf.shape(q)[0]
        # tar, tar, enc_out
        # apply w
        v = self.wv(v)
        k = self.wk(k)
        q = self.wq(q)

        # split
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # calc. attention
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask, wordlen)
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, wordlen=None, block_num=None):

        attn_output, _ = self.mha(x, x, x, mask, training, wordlen, block_num, 3)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             pro_padding_mask, cross_padding_mask, wordlen=None,
             block_num=None):  # tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        attn1, attn_weights_block1 = self.mha1(x, x, x, pro_padding_mask, training, wordlen, block_num, 1)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)  # attn1, x (49, 3000, 768)

        # mixing RNA and protein features
        attn2, attn_weights_block2 = self.mha2(  # v, k, q, mask,
            # (batch_size, target_seq_len, d_model)
            out1, out1, enc_output, cross_padding_mask, training, wordlen, block_num, 2)
        attn2 = self.dropout2(attn2, training=training)  # (49, 105, 768)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        # FFN
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, wordlen=None):
        x = self.dropout(x, training=training)
        x /= tf.cast(self.d_model, tf.float32)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask, wordlen, i)
        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,  # tar, enc_output,
             pro_padding_mask, cross_padding_mask, wordlen=None):  # x = tar previously
        attention_weights = {}
        x = tf.cast(x, tf.float32)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x /= tf.cast(self.d_model, tf.float32)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   pro_padding_mask, cross_padding_mask, wordlen, i)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

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
    # if not os.path.isfile(logfile):
    #     with open(logfile, "w") as f:
    #         pass

    # loss_object = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)
    # loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    # ------------------------------------------------
    # Class & Function
    # ------------------------------------------------

    def loss_function(labellist, predictions):
        if labellist[0] == 0:
            label = [1, 0]
        else:
            label = [0, 1]
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        lossvalue = tf.cast(bce(label, predictions), dtype="float32")
        return lossvalue


    ###############################################
    # 1. read files to 3 lists
    ###############################################
    input_list_list, label_list_list = [], []
    """## **ここからがメイン。**"""

    transformer = Transformer(num_layers, d_model, num_heads, dff,
                              input_vocab_size, target_vocab_size,
                              pe_input=input_vocab_size,
                              pe_target=target_vocab_size,
                              rate=dropout_rate)

    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(1.5e-6, beta_1=0.9, beta_2=0.9999,
                                         epsilon=1e-6)
    # optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
    #                                      epsilon=1e-9)
    # optimizer = tf.keras.optimizers.Adam(0.01, beta_1=0.9, beta_2=0.98,
    #                                      epsilon=1e-2, amsgrad=True)
    # optimizer = tf.keras.optimizers.Nadam()
    # optimizer = tf.keras.optimizers.SGD()

    # optimizer = tf.keras.optimizers.Adadelta(learning_rate, rho=0.95,
    #                                          epsilon=1e-07, name='Adadelta')

    # ----------------------------
    # check point
    # ----------------------------
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if use_checkpoint:
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)


    # get protein vecs
    def get_three_veclists(inp):
        pro_vec_list = []
        rna_vec_list = []
        pro_padding_mask_list = []
        label_list = []
        rna_padding_mask_list = []
        # for item in inp_data:
        item = inp[1]
        try:
            protein_name = PROTNAMES[item[0][0]]
            # protein seqs  pvecs = (number of vectors, 768)
            if protein_name not in tfconfig.testlist:
                return None, None, 0, 0, 0, 0, 0
            pvecs = tf.convert_to_tensor(protein_seq_dict[protein_name][0], dtype=tf.float32)
            zeros = tf.convert_to_tensor([[0] * 768] * (3000 - len(pvecs)), dtype=tf.float32)
            padded_vecs = tf.concat([pvecs, zeros], 0)
            pro_vec_list.append(padded_vecs)

            # p mask          seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
            p_paddingmask = tf.concat([[0]*pvecs.shape[0], [1]*(3000 - pvecs.shape[0])], 0)
            pro_padding_mask_list.append(tf.cast(p_paddingmask, dtype="float32"))

            # rna seqs
            rvecs = tf.convert_to_tensor(item[1], dtype=tf.float32)
            zeros = tf.convert_to_tensor([[0] * 768] * (3000 - len(rvecs)), dtype=tf.float32)
            padded_vecs = tf.concat([rvecs, zeros], 0)
            rna_vec_list.append(padded_vecs)

            # r mask
            r_padding_mask = tf.concat([[0]*102, [1]*2898], 0)
            rna_padding_mask_list.append(tf.cast(r_padding_mask, dtype="float32"))

            # cross mask
            z = tf.convert_to_tensor([0] * 3000, dtype="int32")
            cross_mask = tf.repeat([p_paddingmask, z], repeats=[102, 2898], axis=0)

            # label list
            label = item[2][0]
            label_list.append(label)

        except KeyError as e:
            # tf.print(e)
            return None, None, None, None, None, None, None

        # shape masks
        pro_padding_mask_list = tf.cast(pro_padding_mask_list, dtype="float32")
        pro_padding_mask_list = pro_padding_mask_list[:, tf.newaxis, tf.newaxis, :]
        rna_padding_mask_list = tf.cast(rna_padding_mask_list, dtype="float32")
        rna_padding_mask_list = rna_padding_mask_list[:, tf.newaxis, tf.newaxis, :]

        return [protein_name, tf.convert_to_tensor(pro_vec_list), tf.convert_to_tensor(rna_vec_list), label_list, pro_padding_mask_list, rna_padding_mask_list, cross_mask]


    # @tf.function(input_signature=train_step_signature)
    @tf.function()
    def inference(inp0, m, dlist):  # auc, allist = inference(item, auc, alllist)
        # get protein seqs, and make masks
        # inp : batch data from file
        pro_name, tar_inp, inp, labellist, pro_padding, rna_padding, cross_mask = get_three_veclists(inp0)  # protein features vecs

        if tar_inp is None:
            return m, dlist
        # tar_inp : a bunch of tensors whitch are protein vecs
        # inp : rna vec tensors
        enc_padding_mask = rna_padding
        dec_padding_mask = pro_padding
        # enc_padding_mask, combined_mask, dec_padding_mask = create_masks_auth(inp, tar_inp, pro_effective_vec_len_list)
        # with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,  # "call" is called here
                                                 True,
                                                 enc_padding_mask,
                                                 dec_padding_mask,
                                                 dec_padding_mask)
        # loss = loss_function(labellist, predictions)
        m.update_state(labellist, [tf.nn.softmax(predictions).numpy()[1]])
        dlist.append(np.array([pro_name, labellist, [tf.nn.softmax(predictions).numpy()[1]]]))
        if tfconfig.do_attention:  # make attention data for only one RNA seq and protein seq
            # here save attention data and else for later analysis
            file_id = tfconfig.group_to_ignore  # protein group
            data_id = inp0[0]  # what is this ???
            attention = np.array([pro_name, _["decoder_layer1_block2"]])
            np.save(f"{BASE_PATH}attention/{file_id}_{data_id}_{pro_name}", attention)
        return m, dlist


    # make instances for evaluating optimization
    auc = tf.keras.metrics.AUC()
    alllist = []
    auc.reset_states()

    for batch_count in range(tfconfig.max_files_test):
        # 1 BATCH(FILE)
        train_array = np.load(f"{RNA_SEQ_FILE}{batch_count}.npy", allow_pickle=True)  # (50, 3,)
        for item in enumerate(train_array):  # item (3,)
            # 1 RECORD
            auc, alllist = inference(item, auc, alllist)
            # batch_loss(loss)
        # tf.print(f"batch: {batch_loss.result():.6f}")
        # if batch_count % 10 == 0:
        #     ckpt_manager.save()
    if not tfconfig.do_attention:
        np.save(tfconfig.auc_datapath, np.array(alllist))
        tf.print(f"AUROC:{auc.result().numpy()}")
        tf.print("len(tfconfig.testlist)")
        tf.print(len(tfconfig.testlist))
        tf.print("tfconfig.testlist")
        tf.print(tfconfig.testlist)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    # command example :  " . unknown_protein_predict.py protein_group_ignored trainedfiles testfiles"
    # command example :  " . unknown_protein_predict.py 0 30 50"
    prot_group_to_ignore = int(sys.argv[1])  # get protein group number to ignore
    tfconfig = Tfconfig()
    tfconfig.max_files_trained = int(sys.argv[2])
    tfconfig.max_files_test = int(sys.argv[3])
    if int(sys.argv[4]) == 1:
        tfconfig.do_attention = True
        tfconfig.max_files_test = 1
    tfconfig.group_to_ignore = prot_group_to_ignore
    tfconfig.update()
    opt_trfmr(tfconfig)

