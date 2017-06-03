from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import numpy as np

from util import log
from pprint import pprint

from input_ops import create_input_ops, check_data_id

import os
import time
import numpy as np
import tensorflow as tf
import h5py

class PoseEvalManager(object):

    def __init__(self):
        # collection of batches (not flattened)
        self._ids = []
        self._predictions = []
        self._groundtruths = []

    def add_batch(self, id, prediction, groundtruth):
        assert prediction.shape == groundtruth.shape

        # for now, store them all (as a list of minibatch chunks)
        self._ids.append(id)
        self._predictions.append(prediction)
        self._groundtruths.append(groundtruth)

    def compute_l2error(self, pred, gt):
        errors = (pred - gt) ** 2
        batch_size = pred.shape[0]
        errors = (errors.reshape([batch_size, -1]).sum(axis=1) / np.prod(pred.shape[1:])) ** 0.5
        assert errors.shape == (batch_size,)
        return errors
        #return np.average(errors)

    def report(self):
        # report L2 loss
        log.info("Computing scores...")
        score = {}
        score['l2_loss'] = []

        for id, pred, gt in zip(self._ids, self._predictions, self._groundtruths):
            score['l2_loss'].extend(self.compute_l2error(pred, gt))

        avg_l2loss = np.average(score['l2_loss'])
        log.infov("Average L2 loss : %.5f", avg_l2loss)


    def dump_result(self, filename):
        log.infov("Dumping prediction result into %s ...", filename)
        f.h5py.File(filename, 'w')
        f['test'] = np.concatenate(self._predictions)
        f['test_gt'] = np.concatenate(self._groundtruths)
        f['id'] = str(np.concatenate(self._ids))
        log.info("Dumping prediction done.")

class Evaler(object):

    @staticmethod
    def get_model_class(model_name):
        if model_name == 'MLP':
            from model import Model
        else:
            return ValueError(model_name)
        return Model


    def __init__(self,
                 config,
                 dataset):
        self.config = config
        self.train_dir = config.train_dir
        self.output_file = config.output_file
        log.info("self.train_dir = %s", self.train_dir)

        # --- input ops ---
        self.batch_size = config.batch_size

        self.dataset = dataset

        check_data_id(dataset, config.data_id)
        _, self.batch = create_input_ops(dataset, self.batch_size,
                                         data_id=config.data_id,
                                         is_training=False,
                                         shuffle=False)

        # --- create model ---
        Model = self.get_model_class(config.model)
        self.model = Model(config)

        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
        self.step_op = tf.no_op(name='step_no_op')

        tf.set_random_seed(1234)

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
        )
        self.session = tf.Session(config=session_config)

        # --- checkpoint and monitoring ---
        self.saver = tf.train.Saver(max_to_keep=100)

        self.checkpoint_path = config.checkpoint_path
        if self.checkpoint_path is None and self.train_dir:
            self.checkpoint_path = tf.train.latest_checkpoint(self.train_dir)
        if self.checkpoint_path is None:
            #raise RuntimeError("Either checkpoint_path or train_dir must be given")
            log.warn("No checkpoint is given. Just random initialization :-)")
            self.session.run(tf.global_variables_initializer())
        else:
            log.info("Checkpoint path : %s", self.checkpoint_path)

    def eval_run(self):
        # load checkpoint
        if self.checkpoint_path:
            #import pudb; pudb.set_trace()
            self.saver.restore(self.session, self.checkpoint_path)
            log.info("Loaded from checkpoint!")

        log.infov("Start 1-epoch Inference and Evaluation")

        log.info("# of examples = %d", len(self.dataset))
        length_dataset = len(self.dataset)
        if self.config.max_examples:
            length_dataset = min(self.config.max_examples, length_dataset)
            log.infov("Limiting the number of examples to %d ...", length_dataset)

        #max_steps = int(length_dataset / self.batch_size) + 1
        max_steps = 1
        #assert len(self.dataset) % self.batch_size == 0
        log.info("max_steps = %d", max_steps)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.session,
                                               coord=coord, start=True)

        evaler = PoseEvalManager()
        try:
            for s in xrange(max_steps):
                step, loss, step_time, batch_chunk, prediction_pred, prediction_gt = \
                    self.run_single_step(self.batch)
                self.log_step_message(s, loss, step_time)
                #loss_test, prediction_test = self.run_test(self.batch_test, is_train=False)
                evaler.add_batch(batch_chunk['id'], prediction_pred, prediction_gt)

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        try:
            coord.join(threads, stop_grace_period_secs=3)
        except RuntimeError as e:
            log.warn(str(e)) # just simply ignore as of now

        evaler.report()
        log.infov("Evaluation complete.")

        if self.config.output_file:
            evaler.dump_result(self.config.output_file)

    def run_single_step(self, batch):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        [step, loss, all_preds, all_targets, _] = self.session.run(
            [self.global_step, self.model.total_loss, self.model.all_preds, self.model.all_targets, self.step_op],
            feed_dict=self.model.get_feed_dict(batch_chunk)
        )

        _end_time = time.time()

        return step, loss, (_end_time - _start_time), batch_chunk, all_preds, all_targets

    def run_test(self, batch, is_train=False):

        batch_chunk = self.session.run(batch)

        [loss, all_preds, all_targets] = self.session.run(
            [self.model.total_loss, self.model.all_preds, self.model.all_targets],
            feed_dict=self.model.get_feed_dict(batch_chunk)
        )

        #if self.summary_writer:
        #    self.summary_writer.add_summary(summary, global_step=step)
        return loss, all_preds, all_targets

    def log_step_message(self, step, loss, step_time, is_train=False):
        if step_time == 0: step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                "batch total-loss (test): {test_loss:.5f} " +
                "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) "
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step=step,
                         test_loss=loss,
                         sec_per_batch=step_time,
                         instance_per_sec=self.batch_size / step_time,
                         )
               )

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model', type=str, default='MLP')
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist'])
    parser.add_argument('--data_id', nargs='*', default=None)
    parser.add_argument('--max_examples', type=int, default=None)
    """
    parser.add_argument('--input_height', type=int, default=28)
    parser.add_argument('--input_width', type=int, default=28)
    parser.add_argument('--num_class', type=int, default=10)
    """
    config = parser.parse_args()

    if config.dataset == 'mnist':
        from mnist_dataset import create_default_splits
        config.input_height = 28
        config.input_width = 28
        config.num_class = 10
        dataset_train, dataset_test = create_default_splits()
    else:
        raise ValueError(config.dataset)

    evaler = Evaler(config, dataset_test)

    log.warning("dataset: %s", config.dataset)
    evaler.eval_run()

if __name__ == '__main__':
    main()
