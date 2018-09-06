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


class TransE(object):

    def __init__(self, dataset, model_path, **kwargs):
        self.ds = dataset
        self.model_path = model_path
        embedding_size = kwargs.get('embedding_size', 256)
        bound = 6 / math.sqrt(embedding_size)
        self.margin_value = kwargs.get('margin', 1.0)
        self.embedding_size = embedding_size
        with tf.variable_scope('embeddings'):
            self.node_embeddings = tf.get_variable(
                "node_embeddings", [self.ds.nb_nodes, embedding_size],
                initializer=tf.random_uniform_initializer(minval=-bound, maxval=bound))
            self.relation_embeddings = tf.get_variable(
                "relation_embeddings", [self.ds.nb_relations, embedding_size],
                initializer=tf.random_uniform_initializer(minval=-bound, maxval=bound))
        self.pos_triples = tf.placeholder(tf.int32, shape=(None, 3))
        self.neg_triples = tf.placeholder(tf.int32, shape=(None, 3))
        self.margin = tf.placeholder(tf.float32, shape=(None))
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
        pt = tf.nn.embedding_lookup(self.node_embeddings, self.pos_triples[:, 2])
        
        nh = tf.nn.embedding_lookup(self.node_embeddings, self.neg_triples[:, 0])
        nl = tf.nn.embedding_lookup(self.relation_embeddings, self.neg_triples[:, 1])
        nt = tf.nn.embedding_lookup(self.node_embeddings, self.neg_triples[:, 2])

        pos = tf.subtract(tf.add(ph, pl), pt)
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

        self._loss = tf.norm(pos, axis=1) - tf.norm(neg, axis=1) + self.margin
        self._loss = tf.reduce_sum(tf.nn.relu(self._loss))

        return self._loss

    def train(self, sess, batch_size=256, epochs=100):

        train_generator = Generator(
            self.ds.train_triples,
            self.ds.train_triple_set,
            self.ds.nb_nodes,
            batch_size)
        
        valid_generator = Generator(
            self.ds.valid_triples,
            self.ds.train_triple_set,
            self.ds.nb_nodes,
            batch_size)

        min_valid_loss = 1000000.0
        
        for epoch in range(epochs):
            total_loss = 0
            print('Epoch', (epoch + 1))
            for pt, nt in train_generator:
                feed_dict = {
                    self.pos_triples: pt,
                    self.neg_triples: nt, self.margin: self.margin_value}
                loss = sess.run(self.loss, feed_dict)
                total_loss += loss
                sess.run(self.optimizer, feed_dict)
            print('Train loss: ', total_loss / self.ds.train_size)

            valid_loss = 0
            for pt, nt in valid_generator:
                feed_dict = {
                    self.pos_triples: pt,
                    self.neg_triples: nt,
                    self.margin: self.margin_value}
                loss = sess.run(self.loss, feed_dict)
                valid_loss += loss
            valid_loss /= self.ds.valid_size
            print('Valid loss: ', valid_loss)
            if min_valid_loss > valid_loss:
                min_valid_loss = valid_loss
                print('Saving best model to', self.model_path)
                self.saver.save(sess, self.model_path)
            else:
                print('Validation loss did not improve')


    @property
    def eval_predictions(self):
        if self._eval_predictions is not None:
            return self._eval_predictions
        h = tf.nn.embedding_lookup(self.node_embeddings, self.triple[0])
        l = tf.nn.embedding_lookup(self.relation_embeddings, self.triple[1])
        t = tf.nn.embedding_lookup(self.node_embeddings, self.triple[2])

        # L2 norm
        dist_head = tf.reduce_sum(tf.square(self.node_embeddings + l - t), axis=[1])
        dist_tail = tf.reduce_sum(tf.square(h + l - self.node_embeddings), axis=[1])

        _, index_head = tf.nn.top_k(dist_head, k=self.ds.nb_nodes)
        index_head = tf.reverse(index_head, axis=[0])
        _, index_tail = tf.nn.top_k(dist_tail, k=self.ds.nb_nodes)
        index_tail = tf.reverse(index_tail, axis=[0])

        self._eval_predictions = index_head, index_tail
        return self._eval_predictions
        
    def evaluate(self, sess):
        print('Restoring variables from', self.model_path)
        self.saver.restore(sess, self.model_path)
        
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
        
