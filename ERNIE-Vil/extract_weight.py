from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from numpy.core.arrayprint import printoptions

os.environ['FLAGS_cpu_deterministic'] = 'True'
os.environ['FLAGS_cudnn_deterministic'] = 'True'

import pdb
import sys
import time
import datetime
import argparse
import multiprocessing
import json
import random
# from numpy import random
import numpy as np
import pandas as pd

from reader.meme_finetuning import MemeDataJointReader
from model.ernie_vil import ErnieVilModel, ErnieVilConfig
from optim.optimization import optimization
from utils.args import print_arguments
from utils.init import init_checkpoint, init_pretraining_params
from args.finetune_args import parser
from loguru import logger
from sklearn.metrics import roc_auc_score, accuracy_score

import paddle
import paddle.fluid as fluid

args = parser.parse_args()


def create_vcr_model(pyreader_name, ernie_config, task_group, is_prediction=False, seed=0):
    """
        create model arc for vcr tasks
    """
    shapes = [[-1, args.max_seq_len, 1],    #src_id 
             [-1, args.max_seq_len, 1],    #pos_id
             [-1, args.max_seq_len, 1],    #sent_id
             [-1, args.max_seq_len, 1],    #task_id
             [-1, args.max_seq_len, 1],    #input_mask
             [-1, args.max_img_len, args.feature_size],  #image_embedding
             [-1, args.max_img_len, 5],     #image_loc
             [-1, args.max_img_len, 1],    #image_mask
             [-1, 1],     #labels
             [-1, 1],     #qids
             [],          #task_index
             [-1, 1],     #binary_labels
             ]
    dtypes = ['int64', 'int64', 'int64', 'int64', 'float32', 'float32', 'float32', 'float32', 
                       'int64', 'int64', 'int64', 'float32']
    lod_levels = [0] * len(dtypes)

    for _ in task_group:
        shapes.append([])
        dtypes.append('float')
        lod_levels.append(0)

    pyreader = fluid.layers.py_reader(
        capacity=30,
        shapes=shapes,
        dtypes=dtypes,
        lod_levels=lod_levels,
        name=pyreader_name,
        use_double_buffer=False)
    
    inputs = fluid.layers.read_file(pyreader)
    src_ids, pos_ids, sent_ids, task_ids, input_mask, image_embeddings, \
         image_loc, image_mask, labels, q_ids, task_index, binary_labels = inputs[: 12]

    ernie_vil = ErnieVilModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        task_ids=task_ids,
        input_mask=input_mask,
        image_embeddings=image_embeddings,
        image_loc=image_loc,
        input_image_mask=image_mask,
        config=ernie_config,
        seed=None,
        init_seed=seed,
        )

    h_cls, h_img = ernie_vil.get_pooled_output()
    task_conf = task_group[0]
    fusion_method = task_conf["fusion_method"]
    fusion_fea = ernie_vil.get_match_score(text=h_cls, image=h_img,         \
                                           dropout_rate=task_conf["dropout_rate"],
                                           mode=fusion_method)
    if is_prediction:
        num_choice = int(task_conf['num_choice'])
        task_name = task_conf.get('task_prefix', 'vcr')
        score = fluid.layers.fc(fusion_fea, 2,
                                param_attr = fluid.ParamAttr(name = task_name + "_fc.w_0",
                                initializer = fluid.initializer.TruncatedNormal(scale = 0.02, seed=seed)),
                                bias_attr = task_name + "_fc.b_0")
        # score = fluid.layers.reshape(score, shape = [-1, num_choice])
        _loss, _softmax = fluid.layers.softmax_with_cross_entropy(logits = score,
                                                                  label = labels, return_softmax = True)
        _acc = fluid.layers.accuracy(input = _softmax, label = labels)
        pred = fluid.layers.argmax(score, axis = 1)
        mean_loss = fluid.layers.mean(_loss)
        task_vars = [mean_loss, _acc, pred, q_ids, labels, _softmax]
        for var in task_vars:
            var.persistable = True
        return pyreader, task_vars
    else:
        start_ind = 12
        mean_loss = fluid.layers.zeros(shape = [1], dtype = 'float32')
        mean_acc = fluid.layers.zeros(shape = [1], dtype = 'float32')
        for task_conf in task_group:
            task_weight = inputs[start_ind]
            start_ind += 1
            num_choice = int(task_conf['num_choice'])
            task_name = task_conf.get('task_prefix', 'vcr')
            score = fluid.layers.fc(fusion_fea, 2,
                                    param_attr = fluid.ParamAttr(name = task_name + "_fc.w_0",
                                    initializer = fluid.initializer.TruncatedNormal(scale = 0.02, seed=seed)),
                                    bias_attr = task_name + "_fc.b_0")

            _loss = fluid.layers.softmax_with_cross_entropy(score, labels)
            # _loss = fluid.layers.pow(_loss, 2)
            # _loss = fluid.layers.sigmoid_cross_entropy_with_logits(score,
            #                                                         binary_labels, name = "cross_entropy_loss")
            # tmp_score = fluid.layers.reshape(score, shape = [-1, num_choice])
            
            _softmax = fluid.layers.softmax(score)
            _acc = fluid.layers.accuracy(input = _softmax, label = labels)
            _mean_loss = fluid.layers.mean(_loss)
            mean_loss += _mean_loss * task_weight
            mean_acc += _acc * task_weight
        task_vars = [fluid.layers.reduce_mean(mean_loss), mean_acc]
        for var in task_vars:
            var.persistable = True

        return pyreader, task_vars


def predict_wrapper(args,
                    exe,
                    ernie_config,
                    task_group,
                    test_prog=None,
                    pyreader=None,
                    graph_vars=None):
    """Context to do validation.
    """
    data_reader = MemeDataJointReader(
        task_group,
        split=args.test_split,
        vocab_path=args.vocab_path,
        is_test=True,
        shuffle=False,
        batch_size=args.batch_size,
        epoch=args.epoch,
        random_seed=args.seed,
        balance_cls=False)
    if args.do_test:
        assert args.init_checkpoint is not None, "[FATAL] Please use --init_checkpoint '/path/to/checkpoints' \
                                                  to specify you pretrained model checkpoints"

        init_pretraining_params(exe, args.init_checkpoint, test_prog)
        print(("testing on %s %s split") % (args.task_name, args.test_split))

    def predict(exe=exe, pyreader=pyreader):
        """
            inference for downstream tasks
        """
        pyreader.decorate_tensor_provider(data_reader.data_generator())
        pyreader.start()

        cost = 0
        appear_step = 0
        task_acc = {}
        task_steps = {}
        steps = 0
        case_f1 = 0
        appear_f1 = 0
        time_begin = time.time()
        task_name_list = [v.name for v in graph_vars]
        fetch_list = task_name_list

        print('task name list : ', task_name_list)
        sum_acc = 0
        res_arr = []
        res_csv = []
        
        label_list = []
        pred_probs = []
        
        while True:
            try:
                outputs = exe.run(fetch_list=fetch_list, program=test_prog)
                each_acc = outputs[1][0]
                preds = np.reshape(outputs[2], [-1])
                qids = np.reshape(outputs[3], [-1])
                labels = np.reshape(outputs[4], [-1])
                scores = np.reshape(outputs[5], [-1, 2])
                sum_acc += each_acc
                steps += 1
                if steps % 10 == 0:
                    print('cur_step:', steps, 'cur_acc:', sum_acc / steps)
                
                # format_result(res_arr, qids.tolist(), preds.tolist(), labels.tolist(), scores.tolist())
                for qid, prob in zip(qids, scores[:, 1]):
                    res_csv.append({
                        'id': int(qid),
                        'proba': float(prob),
                        'label': int(float(prob) > 0.5),
                    })
                
                for score, label in zip(scores.tolist(), labels.tolist()):
                    pred_probs.append(score[1])
                    label_list.append(label)
            except fluid.core.EOFException:
                pyreader.reset()
                break

        used_time = time.time() - time_begin

        with open(args.result_file, "w") as f:
            for r in res_arr:
                f.write(r + "\n")

        if args.test_split == 'test':
            pd.DataFrame.from_dict(res_csv).to_csv(args.result_file + '.csv', index=False)
            logger.info(f"Save {args.result_file}")

        print(f'processed {len(label_list)} samples')
        print("average_acc:", sum_acc / steps)
        if args.test_split == 'val':
            print("roc auc: ", roc_auc_score(label_list, pred_probs))

        ret = {}
        ret["acc"] = "acc: %f" % (sum_acc / steps)  
        for item in ret:
            try:
                ret[item] = ret[item].split(':')[-1]
            except:
                pass
        return ret
    return predict


def main(args):
    """
       Main func for downstream tasks
    """
    print("finetuning tasks start")
    ernie_config = ErnieVilConfig(args.ernie_config_path)
    # ernie_config.print_config()
    # import pdb; pdb.set_trace()
    # paddle.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(args.task_group_json) as f:
        task_group = json.load(f)
        print('task: ', task_group)

    startup_prog = fluid.Program()
    startup_prog.random_seed = args.seed
    
    if args.do_test:
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_pyreader, model_outputs  = create_vcr_model(
                    pyreader_name='test_reader', ernie_config=ernie_config, task_group=task_group, is_prediction=True, seed=args.seed)
                total_loss = model_outputs[0]

        test_prog = test_prog.clone(for_test=True)
    
    if args.use_gpu:
        gpu_id = 0
        if os.getenv("FLAGS_selected_gpus"):
            gpu_id = int(os.getenv("FLAGS_selected_gpus"))
    place = fluid.CUDAPlace(gpu_id) if args.use_gpu else fluid.CPUPlace()

    print("theoretical memory usage: ")
    if args.do_test:
        print(fluid.contrib.memory_usage(
            program=test_prog, batch_size=args.batch_size))

    nccl2_num_trainers = 1
    nccl2_trainer_id = 0
    print("args.is_distributed:", args.is_distributed)
    trainer_id = 0

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    exec_strategy = fluid.ExecutionStrategy()
    if args.use_fast_executor:
        exec_strategy.use_experimental_executor = True
    exec_strategy.num_threads = 1
    
    exec_strategy.num_iteration_per_drop_scope = min(10, args.skip_steps)

    build_strategy = fluid.compiler.BuildStrategy()
    build_strategy.fuse_all_reduce_ops = False

    if args.use_fuse:
        build_strategy.fuse_all_reduce_ops = True

    predict = predict_wrapper(
        args,
        exe,
        ernie_config,
        task_group,
        test_prog=test_prog,
        pyreader=test_pyreader,
        graph_vars=model_outputs)
    # result = predict()
    print('-' * 100)
    
    param_list = exe.run(fetch_list=test_prog.all_parameters())
    param_names = [p.name for p in test_prog.all_parameters()]
    for name, param in zip(param_names, param_list):
        print(name, param.shape)
    print(len(param_list))

    save_path = os.path.join(args.checkpoints, "export_np")
    np.savez(
        save_path,
        **{name: param for name, param in zip(param_names, param_list)}
    )


if __name__ == '__main__':
    with logger.catch():
        # print_arguments(args)
        main(args)