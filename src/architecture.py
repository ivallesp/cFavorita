import tensorflow as tf

class NameSpacer:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Seq2Seq:
    def __init__(self, n_numerical_features, categorical_cardinalities, embedding_sizes, n_output_ts, name="S2S"):
        self.name = name

        self.n_numerical_features = n_numerical_features
        self.embedding_sizes = embedding_sizes
        self.categorical_cardinalities = categorical_cardinalities
        self.n_output_ts = n_output_ts
        self.n_categorical_features = len(categorical_cardinalities)
        self.optimizer_function = tf.train.AdamOptimizer(learning_rate=0.001)
        self.define_computation_graph()

        # Aliases
        self.ph = self.placeholders
        self.op = self.optimizers
        self.summ = self.summaries

    def define_computation_graph(self):
        # Reset graph
        tf.reset_default_graph()
        self.placeholders = NameSpacer(**self.define_placeholders())
        self.core_model = NameSpacer(**self.define_core_model())
        self.losses = NameSpacer(**self.define_losses())
        self.optimizers = NameSpacer(**self.define_optimizers())
        self.summaries = NameSpacer(**self.define_summaries())

    def define_placeholders(self):
        placeholders = dict()
        with tf.variable_scope("Placeholders"):
            for var, cardinality in self.categorical_cardinalities.items():
                ph_name = "cat_{}".format(var)
                placeholders[ph_name] = tf.placeholder(dtype=tf.int32, shape=(None, None, 1), name=ph_name)

            placeholders["numerical_feats"] = tf.placeholder(dtype=tf.float32, shape=(None, None,
                                                                                      self.n_numerical_features))

            placeholders["target"] = tf.placeholder(dtype=tf.float32, shape=(None, self.n_output_ts, 1))
            placeholders["loss_dev"] = tf.placeholder(dtype=tf.float32, shape=None, name="loss_dev_manual")
            placeholders["mape_dev"] = tf.placeholder(dtype=tf.float32, shape=None, name="mape_dev_manual")
            placeholders["mape_train"] = tf.placeholder(dtype=tf.float32, shape=None, name="mape_train_manual")

        return placeholders

    def define_core_model(self):
        with tf.variable_scope("Core_Model"):
            embedded_repr = []
            for var, cardinality in self.categorical_cardinalities.items():
                emb_size = self.embedding_sizes[var]
                ph_name = "cat_{}".format(var)
                emb_mat = tf.get_variable(name="emb_mat_"+ph_name, shape=[cardinality, emb_size])
                embedded_repr.append(tf.nn.embedding_lookup(emb_mat, getattr(self.placeholders, ph_name))[:,:,0,:])

            features = tf.concat([self.placeholders.numerical_feats] + embedded_repr, axis=2)

            # Encoder
            cell = tf.keras.layers.LSTMCell(512, name="encoder_cell")
            _, states = tf.nn.dynamic_rnn(cell, features, dtype=tf.float32)

            # Decoder
            go = tf.concat([tf.ones([tf.shape(self.placeholders.target)[0], 1, 512]),
                            tf.zeros([tf.shape(self.placeholders.target)[0], self.n_output_ts-1, 512])], axis=1)
            cell = tf.keras.layers.LSTMCell(512, name="decoder_cell")
            output, _ = tf.nn.dynamic_rnn(cell, go, initial_state=states, dtype=tf.float32)

            output = tf.reshape(output, [-1, 512])
            output = tf.keras.layers.Dense(units=64, activation="relu")(output)
            output = tf.keras.layers.Dense(units=1, activation=None)(output)
            output = tf.reshape(output, [-1, self.n_output_ts, 1])

        return {"output": output}

    def define_losses(self):
        with tf.variable_scope("Losses"):
            loss = tf.losses.mean_squared_error(self.core_model.output, self.placeholders.target)
        return {"loss_mse": loss}

    def define_optimizers(self):
        op = self.optimizer_function.minimize(self.losses.loss_mse)
        return {"op": op}

    def define_summaries(self):
        with tf.variable_scope("Summaries"):
            train_final_scalar_probes = {"loss_mse": tf.squeeze(self.losses.loss_mse)}
            final_performance_scalar = [tf.summary.scalar(k, tf.reduce_mean(v), family=self.name)
                                        for k, v in train_final_scalar_probes.items()]

            train_final_scalar_probes_manual = {"mape_train": self.placeholders.mape_train}
            final_performance_scalar_manual = [tf.summary.scalar(k, tf.reduce_mean(v), family=self.name)
                                        for k, v in train_final_scalar_probes_manual.items()]

            dev_scalar_probes = {"loss_dev": self.placeholders.loss_dev,
                                 "mape_dev": self.placeholders.mape_dev}
            dev_performance_scalar = [tf.summary.scalar(k, v) for k, v in dev_scalar_probes.items()]
        return {"scalar_train_performance": tf.summary.merge(final_performance_scalar),
                "scalar_train_performance_manual":  tf.summary.merge(final_performance_scalar_manual),
                "scalar_dev_performance": tf.summary.merge(dev_performance_scalar)}