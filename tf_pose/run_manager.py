import matplotlib as mpl

mpl.use('Agg')  # training mode, no screen should be open. (It will block training loop)

import argparse
import logging
import json
import os
import sys
import time
from collections import OrderedDict

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from tf_pose.pose_dataset import get_dataflow_batch, DataFlowToQueue, CocoPose
from tf_pose.pose_augment import set_network_input_wh, set_network_scale
from tf_pose.common import get_sample_images, read_imgfile
from tf_pose.networks import get_network, model_wh, get_graph_path
from tf_pose.estimator import TfPoseEstimator


class RunConfig:

    def __init__(self, n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
                 dataset, train_batch_size, test_batch_size, valid_size,
                 opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
                 model_init, init_div_groups, validation_frequency, print_frequency):
        self.n_epochs = n_epochs
        self.init_lr = init_lr
        self.lr_schedule_type = lr_schedule_type
        self.lr_schedule_param = lr_schedule_param

        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.valid_size = valid_size

        self.opt_type = opt_type
        self.opt_param = opt_param
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.no_decay_keys = no_decay_keys

        self.model_init = model_init
        self.init_div_groups = init_div_groups
        self.validation_frequency = validation_frequency
        self.print_frequency = print_frequency

        self._data_provider = None
        self._train_iter, self._valid_iter, self._test_iter = None, None, None

    @property
    def config(self):
        config = {}
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config

    def copy(self):
        return RunConfig(**self.config)

    """ learning rate """

    def _calc_learning_rate(self, epoch, batch=0, nBatch=None):
        if self.lr_schedule_type == 'cosine':
            T_total = self.n_epochs * nBatch
            T_cur = epoch * nBatch + batch
            lr = 0.5 * self.init_lr * (1 + math.cos(math.pi * T_cur / T_total))
        else:
            raise ValueError('do not support: %s' % self.lr_schedule_type)
        return lr

    def adjust_learning_rate(self, optimizer, epoch, batch=0, nBatch=None):
        """ adjust learning of a given optimizer and return the new learning rate """
        new_lr = self._calc_learning_rate(epoch, batch, nBatch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    """ data provider """

    @property
    def data_config(self):
        raise NotImplementedError

    @property
    def data_provider(self):
        if self._data_provider is None:
            if self.dataset == 'imagenet':
                from data_providers.imagenet import ImagenetDataProvider
                self._data_provider = ImagenetDataProvider(**self.data_config)
            elif self.dataset == 'imagenet10':
                from data_providers.imagenet import ImageNet10DataProvider
                self._data_provider = ImageNet10DataProvider(**self.data_config)
            elif self.dataset == 'imagenet100':
                from data_providers.imagenet import ImageNet100DataProvider
                self._data_provider = ImageNet100DataProvider(**self.data_config)
            else:
                raise ValueError('do not support: %s' % self.dataset)
        return self._data_provider

    @data_provider.setter
    def data_provider(self, val):
        self._data_provider = val

    @property
    def train_loader(self):
        return self.data_provider.train

    @property
    def valid_loader(self):
        return self.data_provider.valid

    @property
    def test_loader(self):
        return self.data_provider.test

    @property
    def train_next_batch(self):
        if self._train_iter is None:
            self._train_iter = iter(self.train_loader)
        try:
            data = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            data = next(self._train_iter)
        return data

    @property
    def valid_next_batch(self):
        if self._valid_iter is None:
            self._valid_iter = iter(self.valid_loader)
        try:
            data = next(self._valid_iter)
        except StopIteration:
            self._valid_iter = iter(self.valid_loader)
            data = next(self._valid_iter)
        return data

    @property
    def test_next_batch(self):
        if self._test_iter is None:
            self._test_iter = iter(self.test_loader)
        try:
            data = next(self._test_iter)
        except StopIteration:
            self._test_iter = iter(self.test_loader)
            data = next(self._test_iter)
        return data

    """ optimizer """

    def build_optimizer(self, net_params):
        if self.opt_type == 'sgd':
            opt_param = {} if self.opt_param is None else self.opt_param
            momentum, nesterov = opt_param.get('momentum', 0.9), opt_param.get('nesterov', True)
            if self.no_decay_keys:
                optimizer = torch.optim.SGD([
                    {'params': net_params[0], 'weight_decay': self.weight_decay},
                    {'params': net_params[1], 'weight_decay': 0},
                ], lr=self.init_lr, momentum=momentum, nesterov=nesterov)
            else:
                optimizer = torch.optim.SGD(net_params, self.init_lr, momentum=momentum, nesterov=nesterov,
                                            weight_decay=self.weight_decay)
        else:
            raise NotImplementedError
        return optimizer


def error_and_exit(logger, msg):
    logger.error(msg)
    sys.exit(-1)


class RunManager:

    def __init__(self, path, model=None, cocoyear='2014', coco_dir=None, train_batchsize=64, quant_delay=-1):
        """
        :param path:
        :param model:
        :param coco_dir:
        """

        """ file organization """

        self.path = Path(path)
        self._logs_path = None
        self._save_path = None

        """ logging config """

        self._eval_logger = None
        self._train_logger = None

        """ model config """

        self.model = model or 'mobilenet_v2_1.4'
        self.scale = 4
        if self.model in ['cmu', 'vgg'] or 'mobilenet' in model:
            self.scale = 8

        """ dataset config """

        cocoyear_list = ['2014', '2017']
        if cocoyear not in cocoyear_list:
            raise ValueError('cocoyear should be one of %s' % str(cocoyear_list))

        self.coco_dir = Path(coco_dir) or Path('~/data/coco').expanduser()
        self.annot_dir = self.coco_dir / 'annotations'
        self.val_dir = self.coco_dir / 'val%s' % cocoyear
        self.val_annot = self.annot_dir / 'person_keypoints_val%s.json' % cocoyear

        """ train config """

        self.train_gpus = 4
        self.train_batchsize = train_batchsize
        self.quant_delay = quant_delay
        self.train_resolution = (432, 368)
        self.learning_rate = 0.001
        self.max_epoch = 600

        """ evaluation config """

        self.eval_resolution = (432, 368)
        self.resize_out_ratio = 8.0

        # initialize model (default)
        self.net.init_model(run_config.model_init, run_config.init_div_groups)

        # net info
        self.print_net_info(measure_latency)

        self.criterion = nn.CrossEntropyLoss()
        if self.run_config.no_decay_keys:
            keys = self.run_config.no_decay_keys.split('#')
            self.optimizer = self.run_config.build_optimizer([
                self.net.module.get_parameters(keys, mode='exclude'),  # parameters with weight decay
                self.net.module.get_parameters(keys, mode='include'),  # parameters without weight decay
            ])
        else:
            self.optimizer = self.run_config.build_optimizer(self.net.module.weight_parameters())

    """ save path and log path """

    @staticmethod
    def _get_eval_logger():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
        logger = logging.getLogger('TfPoseEstimator-Video')
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'))
        logger.addHandler(ch)
        return logger

    @staticmethod
    def _get_train_logger():
        logger = logging.getLogger('train')
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'))
        logger.addHandler(ch)
        return logger

    @property
    def eval_logger(self):
        if not self._eval_logger:
            self._eval_logger = self._get_eval_logger()
        return self._eval_logger

    @property
    def train_logger(self):
        if not self._train_logger:
            self._train_logger = self._get_train_logger()
        return self._train_logger

    @property
    def save_path(self):
        if self._save_path is None:
            save_path = os.path.join(self.path, 'checkpoint')
            os.makedirs(save_path, exist_ok=True)
            self._save_path = save_path
        return self._save_path

    @property
    def logs_path(self):
        if self._logs_path is None:
            logs_path = os.path.join(self.path, 'logs')
            os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return self._logs_path

    """ net info """

    def net_flops(self):
        raise NotImplementedError

    def net_latency(self):
        raise NotImplementedError

    def print_net_info(self, measure_latency=None):
        # network architecture
        if self.out_log:
            print(self.net)

        # parameters
        total_params = count_parameters(self.net)
        if self.out_log:
            print('Total training params: %.2fM' % (total_params / 1e6))
        net_info = {
            'param': '%.2fM' % (total_params / 1e6),
        }

        # flops
        flops = self.net_flops()
        if self.out_log:
            print('Total FLOPs: %.1fM' % (flops / 1e6))
        net_info['flops'] = '%.1fM' % (flops / 1e6)

        # latency
        latency_types = [] if measure_latency is None else measure_latency.split('#')
        for l_type in latency_types:
            latency, measured_latency = self.net_latency(l_type, fast=False, given_net=None)
            if self.out_log:
                print('Estimated %s latency: %.3fms' % (l_type, latency))
            net_info['%s latency' % l_type] = {
                'val': latency,
                'hist': measured_latency
            }
        with open('%s/net_info.txt' % self.logs_path, 'w') as fout:
            fout.write(json.dumps(net_info, indent=4) + '\n')

    """ save and load models """

    def save_model(self, checkpoint=None, is_best=False, model_name=None):
        if checkpoint is None:
            checkpoint = {'state_dict': self.net.module.state_dict()}

        if model_name is None:
            model_name = 'checkpoint.pth.tar'

        checkpoint['dataset'] = self.run_config.dataset  # add `dataset` info to the checkpoint
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        model_path = os.path.join(self.save_path, model_name)
        with open(latest_fname, 'w') as fout:
            fout.write(model_path + '\n')
        torch.save(checkpoint, model_path)

        if is_best:
            best_path = os.path.join(self.save_path, 'model_best.pth.tar')
            torch.save({'state_dict': checkpoint['state_dict']}, best_path)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, 'r') as fin:
                model_fname = fin.readline()
                if model_fname[-1] == '\n':
                    model_fname = model_fname[:-1]
        # noinspection PyBroadException
        try:
            if model_fname is None or not os.path.exists(model_fname):
                model_fname = '%s/checkpoint.pth.tar' % self.save_path
                with open(latest_fname, 'w') as fout:
                    fout.write(model_fname + '\n')
            if self.out_log:
                print("=> loading checkpoint '{}'".format(model_fname))

            if torch.cuda.is_available():
                checkpoint = torch.load(model_fname)
            else:
                checkpoint = torch.load(model_fname, map_location='cpu')

            self.net.module.load_state_dict(checkpoint['state_dict'])
            # set new manual seed
            new_manual_seed = int(time.time())
            torch.manual_seed(new_manual_seed)
            torch.cuda.manual_seed_all(new_manual_seed)
            np.random.seed(new_manual_seed)

            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
            if 'best_acc' in checkpoint:
                self.best_acc = checkpoint['best_acc']
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            if self.out_log:
                print("=> loaded checkpoint '{}'".format(model_fname))
        except Exception:
            if self.out_log:
                print('fail to load checkpoint from %s' % self.save_path)

    def save_config(self, print_info=True):
        """ dump run_config and net_config to the model_folder """
        os.makedirs(self.path, exist_ok=True)
        net_save_path = os.path.join(self.path, 'net.config')
        json.dump(self.net.module.config, open(net_save_path, 'w'), indent=4)
        if print_info:
            print('Network configs dump to %s' % net_save_path)

        run_save_path = os.path.join(self.path, 'run.config')
        json.dump(self.run_config.config, open(run_save_path, 'w'), indent=4)
        if print_info:
            print('Run configs dump to %s' % run_save_path)

    """ train and test """

    def write_log(self, log_str, prefix, should_print=True):
        """ prefix: valid, train, test """
        if prefix in ['valid', 'test']:
            with open(os.path.join(self.logs_path, 'valid_console.txt'), 'a') as fout:
                fout.write(log_str + '\n')
                fout.flush()
        if prefix in ['valid', 'test', 'train']:
            with open(os.path.join(self.logs_path, 'train_console.txt'), 'a') as fout:
                if prefix in ['valid', 'test']:
                    fout.write('=' * 10)
                fout.write(log_str + '\n')
                fout.flush()
        if should_print:
            print(log_str)

    def validate(self, data_idx=-1, multi_scale=False):
        """

        :param resize: if provided, resize images before they are processed. default=0x0. Recommends : 432x368 or 656x368 or 1312x736
        :param resize_out_ratio: if provided, upsample heatmaps before they are post-processed. default=8.0
        :param model: cmu / mobilenet_thin / mobilenet_v2_large
        :param cocoyear:
        :param coco_dir:
        :param data_idx:
        :param multi_scale:
        :return:
        """

        coco_dir = self.coco_dir
        image_dir = self.val_dir
        coco_json_file = self.val_annot

        cocoGt = COCO(coco_json_file)
        catIds = cocoGt.getCatIds(catNms=['person'])
        keys = cocoGt.getImgIds(catIds=catIds)

        eval_size = -1
        if data_idx < 0:
            if eval_size > 0:
                keys = keys[:eval_size]  # only use the first #eval_size elements.
            pass
        else:
            keys = [keys[data_idx]]
        logger.info('validation %s set size=%d' % (coco_json_file, len(keys)))
        write_json = '../etcs/%s_%dx%d_%0.1f.json' % (
            self.model, *self.eval_resolution, self.resize_out_ratio)

        logger.debug('initialization %s : %s' % (model, get_graph_path(model)))
        e = TfPoseEstimator(get_graph_path(model), target_size=model_wh(resize))
        print('FLOPs: ', e.get_flops())

        result = []
        tqdm_keys = tqdm(keys)
        for i, k in enumerate(tqdm_keys):
            img_meta = cocoGt.loadImgs(k)[0]
            img_idx = img_meta['id']

            img_name = os.path.join(image_dir, img_meta['file_name'])
            image = read_imgfile(img_name, None, None)
            if image is None:
                error_and_exit(logger, 'image not found, path=%s' % img_name)

            # inference the image with the specified network
            t = time.time()
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=self.resize_out_ratio)
            elapsed = time.time() - t

            scores = 0
            ann_idx = cocoGt.getAnnIds(imgIds=[img_idx], catIds=[1])
            anns = cocoGt.loadAnns(ann_idx)
            for human in humans:
                item = {
                    'image_id': img_idx,
                    'category_id': 1,
                    'keypoints': write_coco_json(human, img_meta['width'], img_meta['height']),
                    'score': human.score
                }
                result.append(item)
                scores += item['score']

            avg_score = scores / len(humans) if len(humans) > 0 else 0
            tqdm_keys.set_postfix(OrderedDict({'inference time': elapsed, 'score': avg_score}))
            if data_idx >= 0:
                logger.info('score:', k, len(humans), len(anns), avg_score)

                import matplotlib.pyplot as plt

                fig = plt.figure()
                a = fig.add_subplot(2, 3, 1)
                plt.imshow(e.draw_humans(image, humans, True))

                a = fig.add_subplot(2, 3, 2)
                # plt.imshow(cv2.resize(image, (e.heatMat.shape[1], e.heatMat.shape[0])), alpha=0.5)
                tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
                plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
                plt.colorbar()

                tmp2 = e.pafMat.transpose((2, 0, 1))
                tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
                tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

                a = fig.add_subplot(2, 3, 4)
                a.set_title('Vectormap-x')
                # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
                plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
                plt.colorbar()

                a = fig.add_subplot(2, 3, 5)
                a.set_title('Vectormap-y')
                # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
                plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
                plt.colorbar()

                plt.show()

        fp = open(write_json, 'w')
        json.dump(result, fp)
        fp.close()

        cocoDt = cocoGt.loadRes(write_json)
        cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
        cocoEval.params.imgIds = keys
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        print(''.join(["%11.4f |" % x for x in cocoEval.stats]))

        pred = json.load(open(write_json, 'r'))

    def train_one_epoch(self):
        raise NotImplementedError

    def _build_train_graph(self):
        # define input placeholder
        set_network_input_wh(*self.train_resolution)

        set_network_scale(self.scale)
        input_w, input_h = self.train_resolution
        output_w, output_h = input_w // self.scale, input_h // self.scale

        logger = self.train_logger
        logger.info('define model+')
        with tf.device(tf.DeviceSpec(device_type="CPU")):
            input_node = tf.placeholder(tf.float32, shape=(self.train_batchsize, *self.train_resolution, 3),
                                        name='image')
            vectmap_node = tf.placeholder(tf.float32, shape=(self.train_batchsize, output_h, output_w, 38),
                                          name='vectmap')
            heatmap_node = tf.placeholder(tf.float32, shape=(self.train_batchsize, output_h, output_w, 19),
                                          name='heatmap')

            # prepare data
            df = get_dataflow_batch(self.annot_dir, is_train=True,
                                    batchsize=self.train_batchsize,
                                    img_path=self.coco_dir)
            enqueuer = DataFlowToQueue(df, [input_node, heatmap_node, vectmap_node], queue_size=100)
            q_inp, q_heat, q_vect = enqueuer.dequeue()

        """ load validation set """

        df_valid = get_dataflow_batch(self.annot_dir, is_train=False,
                                      batchsize=self.train_batchsize,
                                      img_path=self.coco_dir)
        df_valid.reset_state()
        validation_cache = []
        for images_test, heatmaps, vectmaps in tqdm(df_valid.get_data()):
            validation_cache.append((images_test, heatmaps, vectmaps))
        df_valid.reset_state()
        del df_valid
        df_valid = None

        val_image = get_sample_images(*self.train_resolution)
        logger.debug('tensorboard val image: %d' % len(val_image))
        logger.debug(q_inp)
        logger.debug(q_heat)
        logger.debug(q_vect)

        # define model for multi-gpu
        q_inp_split = tf.split(q_inp, self.gpus)
        q_heat_split = tf.split(q_heat, self.gpus)
        q_vect_split = tf.split(q_vect, self.gpus)

        output_vectmap = []
        output_heatmap = []
        losses = []
        last_losses_l1 = []
        last_losses_l2 = []
        outputs = []
        for gpu_id in range(gpus):
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
                with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
                    net, pretrain_path, last_layer = get_network(self.model, q_inp_split[gpu_id])
                    if checkpoint:
                        pretrain_path = checkpoint
                    vect, heat = net.loss_last()
                    output_vectmap.append(vect)
                    output_heatmap.append(heat)
                    outputs.append(net.get_output())

                    l1s, l2s = net.loss_l1_l2()
                    for idx, (l1, l2) in enumerate(zip(l1s, l2s)):
                        loss_l1 = tf.nn.l2_loss(tf.concat(l1, axis=0) - q_vect_split[gpu_id],
                                                name='loss_l1_stage%d_tower%d' % (idx, gpu_id))
                        loss_l2 = tf.nn.l2_loss(tf.concat(l2, axis=0) - q_heat_split[gpu_id],
                                                name='loss_l2_stage%d_tower%d' % (idx, gpu_id))
                        losses.append(tf.reduce_mean([loss_l1, loss_l2]))

                    last_losses_l1.append(loss_l1)
                    last_losses_l2.append(loss_l2)

        outputs = tf.concat(outputs, axis=0)

        with tf.device(tf.DeviceSpec(device_type="GPU")):
            # define loss
            total_loss = tf.reduce_sum(losses) / self.train_batchsize
            total_loss_ll_paf = tf.reduce_sum(last_losses_l1) / self.train_batchsize
            total_loss_ll_heat = tf.reduce_sum(last_losses_l2) / self.train_batchsize
            total_loss_ll = tf.reduce_sum([total_loss_ll_paf, total_loss_ll_heat])

            # define optimizer
            step_per_epoch = 121745 // self.train_batchsize
            global_step = tf.Variable(0, trainable=False)
            if isinstance(self.learning_rate, float):
                # learning_rate = tf.train.exponential_decay(self.learning_rat, global_step,
                #                                            decay_steps=10000, decay_rate=0.33, staircase=True)
                learning_rate = tf.train.cosine_decay(
                    self.learning_rate, global_step,
                    self.max_epoch * step_per_epoch, alpha=0.0)
            else:
                boundaries = [step_per_epoch * 5 * i for i, _ in range(len(self.learning_rate)) if i > 0]
                learning_rate = tf.train.piecewise_constant(global_step, boundaries, self.learning_rate)

        if self.quant_delay >= 0:
            logger.info('train using quantized mode, delay=%d' % self.quant_delay)
            g = tf.get_default_graph()
            tf.contrib.quantize.create_training_graph(input_graph=g, quant_delay=self.quant_delay)

        # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.0005, momentum=0.9, epsilon=1e-10)
        optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
        # optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.8, use_locking=True, use_nesterov=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss, global_step, colocate_gradients_with_ops=True)
        logger.info('define model-')

        # define summary
        tf.summary.scalar("loss", total_loss)
        tf.summary.scalar("loss_lastlayer", total_loss_ll)
        tf.summary.scalar("loss_lastlayer_paf", total_loss_ll_paf)
        tf.summary.scalar("loss_lastlayer_heat", total_loss_ll_heat)
        tf.summary.scalar("queue_size", enqueuer.size())
        tf.summary.scalar("lr", learning_rate)
        merged_summary_op = tf.summary.merge_all()

        valid_loss = tf.placeholder(tf.float32, shape=[])
        valid_loss_ll = tf.placeholder(tf.float32, shape=[])
        valid_loss_ll_paf = tf.placeholder(tf.float32, shape=[])
        valid_loss_ll_heat = tf.placeholder(tf.float32, shape=[])
        sample_train = tf.placeholder(tf.float32, shape=(4, 640, 640, 3))
        sample_valid = tf.placeholder(tf.float32, shape=(12, 640, 640, 3))

        train_img = tf.summary.image('training sample', sample_train, 4)
        valid_img = tf.summary.image('validation sample', sample_valid, 12)
        valid_loss_t = tf.summary.scalar("loss_valid", valid_loss)
        valid_loss_ll_t = tf.summary.scalar("loss_valid_lastlayer", valid_loss_ll)
        merged_validate_op = tf.summary.merge([train_img, valid_img, valid_loss_t, valid_loss_ll_t])

    def train(self, tag='test', checkpoint=''):

        model = self.model
        datapath = self.annot_dir
        imgpath = self.coco_dir
        modelpath = self.save_path
        logpath = self.logs_path

        self._build_train_graph()

        logger = self.train_logger

        saver = tf.train.Saver(max_to_keep=1000)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:

            """ initialize weights or load from checkpoint """

            logger.info('model weights initialization')
            sess.run(tf.global_variables_initializer())

            checkpoint = checkpoint or self.save_path / 'checkpoint'
            if checkpoint and os.path.isdir(checkpoint):
                logger.info('Restore from checkpoint...')
                # loader = tf.train.Saver(net.restorable_variables())
                # loader.restore(sess, tf.train.latest_checkpoint(checkpoint))
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint))
                logger.info('Restore from checkpoint...Done')
            elif pretrain_path:
                logger.info('Restore pretrained weights... %s' % pretrain_path)
                if '.npy' in pretrain_path:
                    net.load(pretrain_path, sess, False)
                else:
                    try:
                        loader = tf.train.Saver(net.restorable_variables(only_backbone=False))
                        loader.restore(sess, pretrain_path)
                    except:
                        logger.info('Restore only weights in backbone layers.')
                        loader = tf.train.Saver(net.restorable_variables())
                        loader.restore(sess, pretrain_path)
                logger.info('Restore pretrained weights...Done')

            """ prepare summary writer and coordinator """

            logger.info('prepare file writer')
            file_writer = tf.summary.FileWriter(os.path.join(logpath, tag), sess.graph)

            logger.info('prepare coordinator')
            coord = tf.train.Coordinator()
            enqueuer.set_coordinator(coord)
            enqueuer.start()

            """ train model """

            logger.info('Training Started.')
            time_started = time.time()
            last_gs_num = last_gs_num2 = 0
            initial_gs_num = sess.run(global_step)

            last_log_epoch1 = last_log_epoch2 = -1

            while True:
                _, gs_num = sess.run([train_op, global_step])
                curr_epoch = float(gs_num) / step_per_epoch

                if gs_num > step_per_epoch * max_epoch:
                    break

                if gs_num - last_gs_num >= 500:
                    train_loss, train_loss_ll, train_loss_ll_paf, train_loss_ll_heat, lr_val, summary = sess.run(
                        [total_loss, total_loss_ll, total_loss_ll_paf, total_loss_ll_heat, learning_rate,
                         merged_summary_op])

                    # log of training loss / accuracy
                    batch_per_sec = (gs_num - initial_gs_num) / (time.time() - time_started)
                    logger.info(
                        'epoch=%.2f step=%d, %0.4f examples/sec lr=%f, loss=%g, '
                        'loss_ll=%g, loss_ll_paf=%g, loss_ll_heat=%g' % (
                            gs_num / step_per_epoch, gs_num, batch_per_sec * self.train_batchsize, lr_val, train_loss,
                            train_loss_ll, train_loss_ll_paf, train_loss_ll_heat))
                    last_gs_num = gs_num

                    if last_log_epoch1 < curr_epoch:
                        file_writer.add_summary(summary, curr_epoch)
                        last_log_epoch1 = curr_epoch

                if gs_num - last_gs_num2 >= 2000:
                    # save weights
                    saver.save(sess, os.path.join(modelpath, tag, 'model_latest'), global_step=global_step)

                    average_loss = average_loss_ll = average_loss_ll_paf = average_loss_ll_heat = 0
                    total_cnt = 0

                    # log of test accuracy
                    for images_test, heatmaps, vectmaps in validation_cache:
                        lss, lss_ll, lss_ll_paf, lss_ll_heat, vectmap_sample, heatmap_sample = sess.run(
                            [total_loss, total_loss_ll, total_loss_ll_paf, total_loss_ll_heat, output_vectmap,
                             output_heatmap],
                            feed_dict={q_inp: images_test, q_vect: vectmaps, q_heat: heatmaps}
                        )
                        average_loss += lss * len(images_test)
                        average_loss_ll += lss_ll * len(images_test)
                        average_loss_ll_paf += lss_ll_paf * len(images_test)
                        average_loss_ll_heat += lss_ll_heat * len(images_test)
                        total_cnt += len(images_test)

                    logger.info('validation(%d) %s loss=%f, loss_ll=%f, loss_ll_paf=%f, loss_ll_heat=%f' % (
                        total_cnt, tag, average_loss / total_cnt, average_loss_ll / total_cnt,
                        average_loss_ll_paf / total_cnt, average_loss_ll_heat / total_cnt))
                    last_gs_num2 = gs_num

                    sample_image = [enqueuer.last_dp[0][i] for i in range(4)]
                    outputMat = sess.run(
                        outputs,
                        feed_dict={q_inp: np.array((sample_image + val_image) * max(1, (self.train_batchsize // 16)))}
                    )
                    pafMat, heatMat = outputMat[:, :, :, 19:], outputMat[:, :, :, :19]

                    sample_results = []
                    for i in range(len(sample_image)):
                        test_result = CocoPose.display_image(sample_image[i], heatMat[i], pafMat[i], as_numpy=True)
                        test_result = cv2.resize(test_result, (640, 640))
                        test_result = test_result.reshape([640, 640, 3]).astype(float)
                        sample_results.append(test_result)

                    test_results = []
                    for i in range(len(val_image)):
                        test_result = CocoPose.display_image(val_image[i], heatMat[len(sample_image) + i],
                                                             pafMat[len(sample_image) + i], as_numpy=True)
                        test_result = cv2.resize(test_result, (640, 640))
                        test_result = test_result.reshape([640, 640, 3]).astype(float)
                        test_results.append(test_result)

                    # save summary
                    summary = sess.run(merged_validate_op, feed_dict={
                        valid_loss: average_loss / total_cnt,
                        valid_loss_ll: average_loss_ll / total_cnt,
                        valid_loss_ll_paf: average_loss_ll_paf / total_cnt,
                        valid_loss_ll_heat: average_loss_ll_heat / total_cnt,
                        sample_valid: test_results,
                        sample_train: sample_results
                    })
                    if last_log_epoch2 < curr_epoch:
                        file_writer.add_summary(summary, curr_epoch)
                        last_log_epoch2 = curr_epoch

            saver.save(sess, os.path.join(modelpath, tag, 'model'), global_step=global_step)

        logger.info('optimization finished. %f' % (time.time() - time_started))

