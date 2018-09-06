import numpy as np
import tensorflow as tf
import math

class Generator(object):

    def __init__(self, data, triple_set, nb_nodes, batch_size):
        self.data = data
        self.triple_set = triple_set
        self.nb_nodes = nb_nodes
        self.size = len(data)
        self.start = 0
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.start < self.size:
            end = self.start + self.batch_size
            batch = self.data[self.start:end]
            neg_triples = []
            for h, l, t in batch:
                while True:
                    hn = h
                    tn = t
                    if np.random.choice([False, True]):
                        hn = np.random.randint(0, self.nb_nodes)
                    else:
                        tn = np.random.randint(0, self.nb_nodes)
                    if (hn, l, tn) not in self.triple_set:
                        neg_triples.append((hn, l, tn))
                        break
            self.start = end
            return (batch, neg_triples)
        else:
            self.start = 0
            raise StopIteration()


class TransH(object):

    def __init__(self, dataset, **kwargs):
        self.ds = dataset
        embedding_size = kwargs.get('embedding_size', 256)
        bound = 6 / math.sqrt(embedding_size)
        self.margin_value = kwargs.get('margin', 1.0)
        self.eps_value = kwargs.get('eps', 1.0)
        self.C_value = kwargs.get('C', 0.25)
        self.embedding_size = embedding_size
        with tf.variable_scope('embeddings'):
            self.node_embeddings = tf.get_variable(
                "node_embeddings", [self.ds.nb_nodes, embedding_size],
                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.relation_embeddings = tf.get_variable(
                "relation_embeddings", [self.ds.nb_relations, embedding_size],
                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.plane_embeddings = tf.get_variable(
                "plane_embeddings", [self.ds.nb_relations, embedding_size],
                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        
        self.pos_triples = tf.placeholder(tf.int32, shape=(None, 3))
        self.neg_triples = tf.placeholder(tf.int32, shape=(None, 3))
        self.margin = tf.placeholder(tf.float32, shape=(None))
        self.eps = tf.placeholder(tf.float32, shape=(None))
        self.C = tf.placeholder(tf.float32, shape=(None))
        self._predictions = None
        self._optimizer = None
        self._loss = None

        self.triple = tf.placeholder(tf.int32, shape=(3,))
        self._eval_predictions = None

        self.saver = tf.train.Saver()

    @property
    def predictions(self):
        if self._predictions is not None:
            return self._predictions
        ph = tf.nn.embedding_lookup(self.node_embeddings, self.pos_triples[:, 0])
        pl = tf.nn.embedding_lookup(self.relation_embeddings, self.pos_triples[:, 1])
        pw = tf.nn.embedding_lookup(self.plane_embeddings, self.pos_triples[:, 1])
        pt = tf.nn.embedding_lookup(self.node_embeddings, self.pos_triples[:, 2])
        
        nh = tf.nn.embedding_lookup(self.node_embeddings, self.neg_triples[:, 0])
        nl = tf.nn.embedding_lookup(self.relation_embeddings, self.neg_triples[:, 1])
        nw = tf.nn.embedding_lookup(self.plane_embeddings, self.neg_triples[:, 1])
        nt = tf.nn.embedding_lookup(self.node_embeddings, self.neg_triples[:, 2])

        pw = tf.nn.l2_normalize(pw, 1)
        dh = tf.reduce_sum(pw * ph, axis=1, keepdims=True) * pw
        dt = tf.reduce_sum(pw * pt, axis=1, keepdims=True) * pw
        ph = ph - dh
        pt = pt - dt
        pos = (ph + pl) - pt

        nw = tf.nn.l2_normalize(nw, 1)
        dh = tf.reduce_sum(nw * nh, axis=1, keepdims=True) * nw
        dt = tf.reduce_sum(nw * nt, axis=1, keepdims=True) * nw
        nh = nh - dh
        nt = nt - dt
        neg = tf.subtract(tf.add(nh, nl), nt)
        
        self._predictions = (pos, neg)

        return self._predictions

    @property
    def optimizer(self):
        if self._optimizer is not None:
            return self._optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        self._optimizer = optimizer.minimize(self.loss)
        return self._optimizer
    

    @property
    def loss(self):
        if self._loss is not None:
            return self._loss

        pos, neg = self.predictions

        self._loss = tf.reduce_sum(tf.square(pos), axis=1) - tf.reduce_sum(tf.square(neg), axis=1) + self.margin
        self._loss = tf.reduce_sum(tf.nn.relu(self._loss))
        scale_reg = tf.reduce_sum(tf.nn.relu(tf.square(self.node_embeddings) - 1))
        ortho_reg = tf.reduce_sum(tf.multiply(self.plane_embeddings, self.relation_embeddings), axis=1)
        ortho_reg = ortho_reg / tf.reduce_sum(tf.square(self.relation_embeddings), axis=1)
        ortho_reg = tf.reduce_sum(tf.nn.relu(ortho_reg - tf.square(self.eps)))
        self._loss = self._loss + self.C * (scale_reg + ortho_reg)
        return self._loss

    def train(self, sess, batch_size=256, epochs=100):
        train_generator = Generator(
            self.ds.train_triples,
            self.ds.triple_set,
            self.ds.nb_nodes,
            batch_size)
        
        valid_generator = Generator(
            self.ds.valid_triples,
            self.ds.triple_set,
            self.ds.nb_nodes,
            batch_size)

        for epoch in range(epochs):
            total_loss = 0
            print('Epoch', (epoch + 1))
            for pt, nt in train_generator:
                feed_dict = {
                    self.pos_triples: pt,
                    self.neg_triples: nt,
                    self.margin: self.margin_value,
                    self.eps: self.eps_value,
                    self.C: self.C_value}
                loss = sess.run(self.loss, feed_dict)
                total_loss += loss
                sess.run(self.optimizer, feed_dict)
            print('Train loss: ', total_loss / self.ds.train_size)

            valid_loss = 0
            for pt, nt in valid_generator:
                feed_dict = {
                    self.pos_triples: pt,
                    self.neg_triples: nt,
                    self.margin: self.margin_value,
                    self.eps: self.eps_value,
                    self.C: self.C_value}
                loss = sess.run(self.loss, feed_dict)
                valid_loss += loss
            valid_loss /= self.ds.valid_size
            print('Valid loss: ', valid_loss)
            if min_valid_loss > valid_loss:
                min_valid_loss = valid_loss
                print('Saving best model to transe.ckpt')
                self.saver.save(sess, 'transh.ckpt')
            else:
                print('Validation loss did not improve')


    @property
    def eval_predictions(self):
        if self._eval_predictions is not None:
            return self._eval_predictions
        w = tf.nn.embedding_lookup(self.plane_embeddings, self.triple[1])
        w = tf.reshape(w, [1, -1])
        w = tf.nn.l2_normalize(w, 1)
        de = tf.reduce_sum(w * self.node_embeddings, axis=1, keepdims=True) * w
        embeddings = self.node_embeddings - de
        h = tf.nn.embedding_lookup(embeddings, self.triple[0])
        l = tf.nn.embedding_lookup(self.relation_embeddings, self.triple[1])
        t = tf.nn.embedding_lookup(embeddings, self.triple[2])

        
        # L2 norm
        dist_head = tf.reduce_sum(tf.square(embeddings + l - t), axis=[1])
        dist_tail = tf.reduce_sum(tf.square(h + l - embeddings), axis=[1])

        _, index_head = tf.nn.top_k(dist_head, k=self.ds.nb_nodes)
        index_head = tf.reverse(index_head, axis=[0])
        _, index_tail = tf.nn.top_k(dist_tail, k=self.ds.nb_nodes)
        index_tail = tf.reverse(index_tail, axis=[0])

        self._eval_predictions = index_head, index_tail
        return self._eval_predictions
        
    def evaluate(self, sess):
        print('Restoring variables from transe.ckpt')
        self.saver.restore(sess, 'transh.ckpt')

        head_rank_raw = 0
        head_rank_fil = 0
        head_hits_raw = 0
        head_hits_fil = 0
        tail_rank_raw = 0
        tail_rank_fil = 0
        tail_hits_raw = 0
        tail_hits_fil = 0

            
        for i, (h, l, t) in enumerate(self.ds.test_triples):
            index_head, index_tail = sess.run(
                self.eval_predictions, {self.triple: [h, l, t]})
            rank_raw = 0
            rank_fil = 0
            for node in index_head:
                if h == node:
                    break
                rank_raw += 1
                if (node, l, t) not in self.ds.triple_set:
                    rank_fil += 1
            if rank_raw < 10:
                head_hits_raw += 1
            if rank_fil < 10:
                head_hits_fil += 1
            head_rank_raw += rank_raw
            head_rank_fil += rank_fil

            rank_raw = 0
            rank_fil = 0
            for node in index_tail:
                if t == node:
                    break
                rank_raw += 1
                if (h, l, node) not in self.ds.triple_set:
                    rank_fil += 1
            if rank_raw < 10:
                tail_hits_raw += 1
            if rank_fil < 10:
                tail_hits_fil += 1
            tail_rank_raw += rank_raw
            tail_rank_fil += rank_fil
        test_size = len(self.ds.test_triples)
        head_rank_raw /= test_size
        head_rank_fil /= test_size
        head_hits_raw /= test_size
        head_hits_fil /= test_size
        print('Head rank raw: %.0f\tHead rank filtered: %.0f' % (head_rank_raw, head_rank_fil))
        print('Head hits raw: %.2f\tHead hits filtered: %.2f' % (head_hits_raw, head_hits_fil))
        tail_rank_raw /= test_size
        tail_rank_fil /= test_size
        tail_hits_raw /= test_size
        tail_hits_fil /= test_size
        print('Tail rank raw: %.0f\tTail rank filtered: %.0f' % (tail_rank_raw, tail_rank_fil))
        print('Tail hits raw: %.2f\tTail hits filtered: %.2f' % (tail_hits_raw, tail_hits_fil))
        mean_rank_raw = (head_rank_raw + tail_rank_raw) / 2
        mean_rank_fil = (head_rank_fil + tail_rank_fil) / 2
        mean_hits_raw = (head_hits_raw + tail_hits_raw) / 2
        mean_hits_fil = (head_hits_fil + tail_hits_fil) / 2
        print('Mean rank raw: %.0f\tMean rank filtered: %.0f' % (mean_rank_raw, mean_rank_fil))
        print('Mean hits raw: %.2f\tMean hits filtered: %.2f' % (mean_hits_raw, mean_hits_fil))
