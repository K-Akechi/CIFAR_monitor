import tensorflow as tf
import numpy as np
import scipy.misc
import time
from data_utility import *
import models

model_save_path = './vgg19/'
log_save_path = './vgg_logs'
total_epoch = 200
iterations = 500
batch_size = 100


# def run_testing(sess, ep):
#     acc = 0.0
#     loss = 0.0
#     pre_index = 0
#     add = 1000
#     for it in range(10):
#         batch_x = test_x[pre_index:pre_index+add]
#         batch_y = test_y[pre_index:pre_index+add]
#         pre_index = pre_index + add
#         loss_, acc_  = sess.run([cross_entropy, accuracy],feed_dict={x:batch_x, y_:batch_y, keep_prob: 1.0, train_flag: False})
#         loss += loss_ / 10.0
#         acc += acc_ / 10.0
#     summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=loss),
#                             tf.Summary.Value(tag="test_accuracy", simple_value=acc)])
#     return acc, loss, summary


def main(argv=None):
    train_x, train_y, test_x, test_y = prepare_data()
    x = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
    y_ = tf.placeholder(tf.float32, [None, class_num])
    train_flag = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)
    y = models.vgg19(x, train_flag)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        summary_writer = tf.summary.FileWriter(log_save_path, sess.graph)
        restore_dir = tf.train.latest_checkpoint(model_save_path)
        if restore_dir:
            saver.restore(sess, restore_dir)
            print('restore succeed')
        else:
            sess.run(init)

        for ep in range(1, total_epoch + 1):
            pre_index = 0
            train_acc = 0.0
            train_loss = 0.0
            start_time = time.time()

            print("\nepoch %d/%d:" % (ep, total_epoch))

            for it in range(1, iterations + 1):
                batch_x = train_x[pre_index:pre_index + batch_size]
                batch_y = train_y[pre_index:pre_index + batch_size]

                _, batch_loss = sess.run([train_step, cross_entropy],
                                         feed_dict={x: batch_x, y_: batch_y, train_flag: True})
                batch_acc = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, train_flag: True})

                train_loss += batch_loss
                train_acc += batch_acc
                pre_index += batch_size

                if it == iterations:
                    train_loss /= iterations
                    train_acc /= iterations

                    # loss_, acc_ = sess.run([cross_entropy, accuracy], feed_dict={x: batch_x, y_: batch_y, train_flag: True})
                    train_summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=train_loss),
                                                      tf.Summary.Value(tag="train_accuracy", simple_value=train_acc)])

                    val_acc = 0.0
                    val_loss = 0.0
                    val_pre_index = 0
                    add = 1000
                    for val_it in range(10):
                        batch_x = test_x[val_pre_index:val_pre_index + add]
                        batch_y = test_y[val_pre_index:val_pre_index + add]
                        val_pre_index = val_pre_index + add
                        loss_, acc_ = sess.run([cross_entropy, accuracy],
                                               feed_dict={x: batch_x, y_: batch_y, train_flag: False})
                        val_loss += loss_ / 10.0
                        val_acc += acc_ / 10.0
                    test_summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=val_loss),
                                                tf.Summary.Value(tag="test_accuracy", simple_value=val_acc)])
                    # val_acc, val_loss, test_summary = run_testing(sess, ep)

                    summary_writer.add_summary(train_summary, ep)
                    summary_writer.add_summary(test_summary, ep)
                    summary_writer.flush()

                    print(
                        "iteration: %d/%d, cost_time: %ds, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f"
                        % (it, iterations, int(time.time() - start_time), train_loss, train_acc, val_loss, val_acc))
                else:
                    print("iteration: %d/%d, train_loss: %.4f, train_acc: %.4f" % (
                    it, iterations, train_loss / it, train_acc / it), end='\r')

        save_path = saver.save(sess, model_save_path)
        print("Model saved in file: %s" % save_path)


if __name__ == "__main__":
    tf.app.run()
