# -------------------------------------------------------------------
# this code load data for LSTM.
# input : 
# output: 
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import numpy as np
import tensorflow as tf
import os
import datetime
import tensorflow_addons as tfa

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------

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

########################################################################
########################################################################

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
        self.dataset_batch = 1
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
        if self.node_name == "f":
            self.num_of_gpu = self.num_of_node * 4
        elif self.node_name == "q":
            self.num_of_gpu = self.num_of_node
        else:
            print("Node name is incorrect !!!")
        if self.run_on_local == 1:
            self.basepath = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/"
            physical_devices = tf.config.list_physical_devices('CPU')  # 8 GPUs in my setup
            tf.config.set_visible_devices(physical_devices[0], 'CPU')  # Using all GPUs (default behaviour)
        elif self.run_on_local == 0:
            self.basepath = "/gs/hs0/tga-science/kimura/transformer_tape_dnabert/"
            gpus = tf.config.list_physical_devices('GPU')  # 8 GPUs in my setup
            tf.config.set_visible_devices(gpus[0:config.num_of_gpu], 'GPU')  # Using all GPUs (default behaviour)
        if self.training == 0:
            self.max_epoch = 1
            self.training_boolean = False
        else:
            self.training_boolean = True
        if self.data_mode == "all":
            self.data_directory = "data"
        elif self.data_mode == "lnc" or self.data_mode == "lnc_unknown":
            self.data_directory = "data_lncRNA"

        if self.data_mode != "lnc_unknown":
            # /Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/training_data_tokenized/all5_small_tape
            self.training_npz_dir = f"{self.basepath}{self.data_directory}/training_data_tokenized/{self.data_dir_name}/"
        else:
            # /Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data_lncRNA/training_data_tokenized/
            self.training_npz_dir = f"{self.basepath}{self.data_directory}/training_data_tokenized/"

        self.statistical_pot_path = f"{self.basepath}{self.data_directory}/attn_arrays_hb/{self.group_to_ignore}/"  # 1a1t-A_1a1t-B.npz
        self.statistical_pot_pi_path = f"{self.basepath}{self.data_directory}/attn_arrays_pi/{self.group_to_ignore}/"  # 1a1t-A_1a1t-B.npz
        if self.run_on_local == 1:
            self.taskname = f"{task_id}_macpro"
        else:
            self.taskname = f"{task_id}_t3"
        if self.training == 1:
            self.checkpoint_path = f"{self.basepath}chpoint/{self.keyword}/chpt_"
        elif self.training == 0:
            self.checkpoint_path = f"{self.basepath}chpoint/{self.keyword}/chpt_"
        self.checkpoint_dir = f"{self.basepath}chpoint/{self.keyword}/"
        # self.data_for_auc = self.basepath + "/data/for_comparison/CrossRPBmer/"

        return self


def opt_trfmr(tfconfig):
    # ------------------------------------------------
    # functions and else
    # ------------------------------------------------
    def loss_function(label, pred):
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM)
        return bce(label, pred)

    # transformer = Transformer(tfconfig)
    learning_rate = tfconfig.init_lr
    # train_step_signature = [
    #     tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    #     tf.TensorSpec(shape=(None, None), dtype=tf.int32),]

    class CustomModel(tf.keras.Model):

        def __init__(self, tffconfig):
            super(CustomModel, self).__init__()
            self.lstm = tf.keras.Sequential([
                tf.keras.layers.Embedding(
                    input_dim=27,
                    output_dim=64,
                    # Use masking to handle the variable sequence lengths
                    mask_zero=True),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation="sigmoid")
            ])

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
            tfconfig.proid = self.flatten_n_batch(datalist[0])
            tfconfig.protok = self.flatten_n_batch(datalist[1])
            tfconfig.rnatok = self.flatten_n_batch(datalist[2])
            add_to_rna = tf.repeat(tf.repeat(tf.constant([[[21]]], dtype="float32"), 101, axis=2), 250, axis=1)

            tfconfig.rnatok = tf.add(tfconfig.rnatok, add_to_rna)  # 25, 100
            label = self.flatten_n_batch(datalist[3])[0, :, 1:]
            connection = tf.repeat(tf.constant([[[0], [0], [0], [0], [0]]], dtype="float32"), 50, axis=1)
            input_combined = tf.concat([tfconfig.rnatok, connection, tfconfig.protok], axis=2)

            with tf.GradientTape() as tape:
                predictions = self.lstm(input_combined[0])  # prediction, pweight, rweight, p_cr_weight, r_cr_weight
                loss = loss_function(label, predictions)
            loss_avg.update_state(loss)
            gradients = tape.gradient(loss, self.trainable_variables)
            auc.update_state(label, predictions)
            if tfconfig.training == 1:
                trainable_variables = [var for grad, var in zip(gradients, self.lstm.trainable_variables) if
                                       grad is not None]
                gradient = [flat_gradients(grad) for grad in gradients if grad is not None]
                optimizer.apply_gradients(zip(gradient, trainable_variables))
            tfconfig.batch_count += 1
            return {"loss": loss_avg.result(), "auc": auc.result()}


        @tf.function()
        # @tf.function(input_signature=train_step_signature)
        def test_step(self, data_combined, y=None, first_batch=None):

            datalist = [x for x in data_combined]
            tfconfig.proid = self.flatten_n_batch(datalist[0])
            tfconfig.protok = self.flatten_n_batch(datalist[1])
            tfconfig.rnatok = self.flatten_n_batch(datalist[2])
            add_to_rna = tf.repeat(tf.repeat(tf.constant([[[20]]], dtype="float32"), 101, axis=2), 25, axis=1)

            tfconfig.rnatok = tf.add(tfconfig.rnatok, add_to_rna)
            label = self.flatten_n_batch(datalist[3])[0, :, 1:]
            connection = tf.repeat(tf.constant([[[0], [0], [0], [0], [0]]], dtype="float32"), 5, axis=1)
            input_combined = tf.concat([tfconfig.rnatok, connection, tfconfig.protok], axis=2)
            predictions = self.lstm(input_combined[0])  # prediction, pweight, rweight, p_cr_weight, r_cr_weight
            loss = loss_function(label, predictions)
            loss_avg.update_state(loss)
            auc.update_state(label, predictions)
            tfconfig.batch_count += 1
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
        five_dataset = four_dataset.concatenate(tf.data.Dataset.list_files(f"{tfconfig.training_npz_dir}{fold_id}/" + "*.npz"))
        return five_dataset


    def np_load(filename):
        arr = np.load(filename.numpy(), allow_pickle=True)
        proid_tf = tf.convert_to_tensor(arr["proid"], dtype="int16")
        protok_tf = tf.convert_to_tensor(arr["protok"], dtype="float32")
        rnatok_tf = tf.convert_to_tensor(arr["rnatok"], dtype="float32")
        label_tf = tf.convert_to_tensor(arr["label"], dtype="int16")
        return (proid_tf, protok_tf, rnatok_tf, label_tf)


    # ------------------------------------------------
    # callbacks and dataset
    # ------------------------------------------------

    optimizer = tfa.optimizers.RectifiedAdam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9,
                                             weight_decay=1e-5)
    current_time = datetime.datetime.now().strftime("%Y%m%d")
    checkpoint_path = f"{tfconfig.checkpoint_path}"
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

    train_log_dir = f"{tfconfig.basepath}/logs_lstm/{tfconfig.taskname}_{current_time}/"
    ctensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=train_log_dir, histogram_freq=1)
    # tf.debugging.set_log_device_placement(True)

    # get paths
    if tfconfig.data_mode == "lnc_unknown":
        npz_list = get_unknown_npzlist(tfconfig.cv_fold_id)
        train_files = 800
        test_files = 200
        val_files = 0
    else:
        npz_list = tf.data.Dataset.list_files(tfconfig.training_npz_dir + "*.npz")
        train_files = 1000
        test_files = 200
        val_files = 200
    # load data
    if tfconfig.use_TAPE_feature == 1:
        combined_dataset = npz_list.map(lambda x: tf.py_function(func=np_load, inp=[x],
            Tout=[tf.int16, tf.float32, tf.float32, tf.int16]))
    else:
        combined_dataset = npz_list.map(lambda x: tf.py_function(func=np_load, inp=[x],
            Tout=[tf.int16, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int16]))
    # batchfy
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    tfconfig.dataset_batch = 200
    # split into three sets
    combined_dataset_train = combined_dataset.take(int(train_files))
    # combined_dataset_train = combined_dataset_train.repeat(1).batch(dataset_batch)
    combined_dataset_train = combined_dataset_train.repeat(1).batch(tfconfig.dataset_batch)

    combined_dataset_test = combined_dataset.skip(int(train_files)).take(int(test_files))
    # combined_dataset_test = combined_dataset_test.repeat(1).batch(dataset_batch)
    combined_dataset_test = combined_dataset_test.repeat(1).batch(20)
    if tfconfig.data_mode != "lnc_unknown":
        combined_dataset_validation = combined_dataset.skip(int(train_files)).skip(test_files).take(int(val_files))
        combined_dataset_validation = combined_dataset_validation.repeat(1).batch(tfconfig.dataset_batch)
    # combined_dataset_train = combined_dataset_train.repeat(1).batch(1).with_options(options).prefetch(tf.data.AUTOTUNE)
    # combined_dataset_test = combined_dataset_test.repeat(1).batch(1).with_options(options).prefetch(tf.data.AUTOTUNE)
    # run model
    auc = tf.keras.metrics.AUC()
    loss_avg = tf.keras.metrics.Mean(name='train_loss')
    callbacks = [ctensorboard_callback, cp_callback, CustomCallback()]

    # # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
    with strategy.scope():
        optimizer = tfa.optimizers.RectifiedAdam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9, weight_decay=1e-5)
        model = CustomModel(tfconfig)
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=optimizer)
        if tfconfig.usechpoint == 1:
            model.load_weights(tfconfig.checkpoint_path)
    history = model.fit(combined_dataset_train, epochs=100, validation_data=combined_dataset_test, callbacks=callbacks)

        # model.fit(combined_dataset_train, epochs=tfconfig.max_epoch, callbacks=callbacks, validation_data=combined_dataset_test)
    # elif tfconfig.training == 0:
    #     model.evaluate(combined_dataset_test)


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
    parser.add_argument('--two_d_softm_mul_row_count', type=int)
    parser.add_argument('--use_TAPE_feature', type=int)
    parser.add_argument('--only_rna_path', type=int)
    parser.add_argument('--only_protein_path', type=int)
    parser.add_argument('--data_mode', default="all")
    parser.add_argument('--cv_fold_id', type=int, default=None)

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
    config.two_d_softm_mul_row_count = args.two_d_softm_mul_row_count
    config.task_identifier = "node_" + str(config.node_name) + "_nodenum_" + str(config.num_of_node) +  \
                            "_aug_" + str(config.use_attn_augument) + "_twoD_" + str(config.two_d_softm) +\
                            "_headnum_" + str(config.num_heads) + "_initLR_" + str(config.init_lr) +\
                             "_keywrd_" + str(config.keyword) + "_clip_coeff_" + str(config.clip_coeff) +\
                            "_cv_fold_id_" + str(config.cv_fold_id) + \
                            "_datamode_" + str(config.data_mode) + "_use_TAPE_feature_" + str(config.use_TAPE_feature)
    config = config.update(config.task_identifier)
    opt_trfmr(config)
