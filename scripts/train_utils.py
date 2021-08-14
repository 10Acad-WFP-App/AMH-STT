import tensorflow as tf

class CTCLossLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        labels = inputs[0]
        logits = inputs[1]
        label_len = inputs[2]
        logit_len = inputs[3]

        logits_trans = tf.transpose(logits, (1,0,2))
        label_len = tf.reshape(label_len, (-1,))
        logit_len = tf.reshape(logit_len, (-1,))
        loss = tf.reduce_mean(tf.nn.ctc_loss(labels, logits_trans, label_len, logit_len, blank_index=-1))
        # define loss here instead of in compile
        self.add_loss(loss)

        # Decode
        decoded, _ = tf.nn.ctc_greedy_decoder(logits_trans, logit_len)

        # Inaccuracy: label error rate
        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),labels))
        self.add_metric(ler, name='ler', aggregation='mean')

        return logits
