# Source code with the blog post at http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
import helper as w2c
import csv_reader as csv
import tensorflow as tf


def read_inputs():

    def flatten(list):
        """ flattens a list of lists into a single list """
        return [elem for sublist in list for elem in sublist]

    fp = "training_data.csv"
    d = csv.read(fp, [1], 0)
    titles, labels = d[1], d[0]
    #to much memory to use all of the data for my laptop (axel)
    titles = titles[:50]
    labels = labels[:50]

    tmp = labels
    labels = map(lambda s: s.split(), labels)

    users = flatten(map(lambda s: s.split(), tmp))

    _, _, user_index_map, rev_user_index_map = w2c.build_dataset(users)

    words = " ".join(titles).split()

    _, _, vocab, rev_vocab = w2c.build_dataset(words)
    return titles, labels, users, user_index_map, vocab


def make_one_hots(words, dic):
    result = []
    for word in words:
        onehot = [0]*len(dic)
        onehot[dic[word]] = 1
        result.append(onehot)
    return result


def make_one_hot(title, dic):
    """
    :param title: title in a plain string
    :param dic: word to index map
    :return: title but as a list of one hot vectors
    """
    words = title.split()
    words = words[:30]
    max_title_length = 30
    inner = [0]*len(dic)
    result = [inner]*max_title_length
    for index, word in enumerate(words):
        one_hot = [0]*len(dic)
        if word not in dic.keys():
            word = "UNK"
        one_hot[dic[word]] = 1
        result[index] = one_hot
    return result


def label_vector(users, dic):
    """
    :param users: list of all users(strings)
    :return:
    """
    vector = [0]*len(dic)
    for user in users:
        vector[dic[user]] = 1
    return vector


titles, labels, users, user_map, vocab = read_inputs()

titles_vectors = list(map(lambda t: make_one_hot(t, vocab), titles))
label_vectors = list(map(lambda l: label_vector(l, user_map), labels))


data = tf.placeholder(tf.float32, [None, 30, len(vocab)])
target = tf.placeholder(tf.float32, [None, len(label_vectors[0])])

dimensions = 10

cell = tf.nn.rnn_cell.LSTMCell(dimensions, state_is_tuple=True)

val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([dimensions, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))

optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)

#behöver ändras kan ha flera som gillar samma post
mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
p = (sess.run(prediction, {data: titles_vectors}))

batch_size = 10
train_input = titles_vectors
train_output = label_vectors

no_of_batches = 1
epoch = 10
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr += batch_size
        sess.run(minimize, {data: inp, target: out})
    print("epoch :" + str(i))

incorrect = sess.run(error, {data: train_input, target: train_output})
print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
sess.close()
