import tensorflow as tf
import pandas as pd

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
LEARNING_RATE = 0.01
EPOCH_NUM = 15
BATCH_SIZE = 100
LOGS_PATH = '/tmp/tensorflow_logs'


def preprocess_data(path, is_test=False):
    data = pd.read_csv(path, index_col='PassengerId')
    data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    if is_test:
        data = data.replace([None], [0])
    else:
        data = data[pd.notnull(data['Age'])]
        data = data[pd.notnull(data['Embarked'])]
    data.replace(["female", "male"], [0, 1], inplace=True)
    data.replace(["Q", "C", "S"], [0, 1, 2], inplace=True)
    if "Survived" in data:
        data = data[pd.notnull(data['Survived'])]
    data_norm = (data - data.mean()) / (data.max() - data.min())
    return data_norm


def next_batch(df, i=None):
    """

    :param df: pandas dataframe
    :param i: batch index
    :return: (numpy array x, numpy array y)
    """
    if i is None:
        start = 0
        end = df.shape[0]
    else:
        start = BATCH_SIZE * i
        end = BATCH_SIZE * (i + 1)
    result = df[start:end]
    if "Survived" in result:
        batch_ys = pd.get_dummies(result.pop('Survived').values).as_matrix()
        batch_xs = result.as_matrix()
        return batch_xs, batch_ys
    else:
        return result.as_matrix()


def split_dataset(df, test_part=None):
    """
    Split dataframe
    :param test_part: float from 0 to 1
    :param df: pandas dataframe
    :return: (pandas dataframe train, pandas dataframe test)
    """
    length = df.shape[0]
    if test_part is None:
        test_part = 0.15

    test_part = int(length * test_part)

    test_dataset = df[0:test_part]
    training_dataset = df[test_part:]
    return training_dataset, test_dataset


dataset = preprocess_data(TRAIN_PATH)

training_dataset, test_narray = split_dataset(dataset)

x = tf.placeholder(tf.float32, [None, 7], name='InputData')
y = tf.placeholder(tf.float32, [None, 2], name='TargetData')

W = tf.Variable(tf.zeros([7, 2]), name='Weights')
b = tf.Variable(tf.zeros([2]), name='Bias')

with tf.name_scope('Model'):
    pred = tf.nn.softmax(tf.matmul(x, W) + b)

with tf.name_scope('Loss'):
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred + 1e-10), reduction_indices=1))

with tf.name_scope('GDS'):
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

with tf.name_scope('Accuracy'):
    acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

init = tf.global_variables_initializer()

tf.summary.scalar("loss", cost)
tf.summary.scalar("accuracy", acc)
merged_summary = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)

    log_writer = tf.summary.FileWriter(LOGS_PATH, graph=tf.get_default_graph())
    training_dataset_size = training_dataset.shape[0]
    for epoch in range(EPOCH_NUM):
        avg_cost = 0.
        total_batch = int(training_dataset_size / BATCH_SIZE)

        for i in range(total_batch):
            batch_xs, batch_ys = next_batch(training_dataset, i)
            _, c, summary = sess.run([optimizer, cost, merged_summary], feed_dict={x: batch_xs, y: batch_ys})
            log_writer.add_summary(summary, epoch * total_batch + i)
            avg_cost += c / total_batch

        print("Epoch:", '%d' % (epoch + 1), "cost=", "{0}".format(avg_cost))

    test_x, test_y = next_batch(test_narray)
    print("Accuracy:", acc.eval({x: test_x, y: test_y}))

    test_df = preprocess_data(TEST_PATH, is_test=True)
    indexes = test_df.index.values
    test_narray = next_batch(test_df)
    feed_dict = {x: test_narray}
    predict_proba = pred.eval(feed_dict)
    predictions = tf.argmax(predict_proba, dimension=1).eval()

    with open("kaggle.csv", "w") as f:
        f.write("PassengerId,Survived\n")
        for index, prediction in zip(indexes, predictions):
            f.write("{0},{1}\n".format(index, prediction))
