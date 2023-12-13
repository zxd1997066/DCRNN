import argparse
import numpy as np
import os
import sys
import yaml
import torch

from lib.utils import load_graph_data
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor


def run_dcrnn(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.safe_load(f)
        supervisor_config['data']['dataset_dir'] = args.dataset_dir
        supervisor_config['data']['test_batch_size'] = args.batch_size

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        if args.precision == "bfloat16":
            print("---- Use cpu AMP to bf16")
            with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)
                mean_score, outputs = supervisor.evaluate('test', args=args)
        elif args.precision == "float16":
            print("---- Use AMP to fp16")
            if args.device == "cpu":
                with torch.cpu.amp.autocast(enabled=True, dtype=torch.half):
                    supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)
                    mean_score, outputs = supervisor.evaluate('test', args=args)
            elif args.device == "cuda":
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.half):
                    supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)
                    mean_score, outputs = supervisor.evaluate('test', args=args)
        else:
            supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)
            mean_score, outputs = supervisor.evaluate('test', args=args)
        # np.savez_compressed(args.output_filename, **outputs)
        print("MAE : {}".format(mean_score))
        # print('Predictions saved as {}.'.format(args.output_filename))


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_filename', default='data/model/pretrained/METR-LA/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--output_filename', default='data/dcrnn_predictions.npz')
    # extra
    parser.add_argument('--device', default="cpu", type=str, help='cpu, cuda or xpu')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--precision', default="float32", type=str, help='precision')
    parser.add_argument('--channels_last', default=1, type=int, help='Use NHWC or not')
    parser.add_argument('--jit', action='store_true', default=False, help='enable JIT')
    parser.add_argument('--profile', action='store_true', default=False, help='collect timeline')
    parser.add_argument('--num_iter', default=-1, type=int, help='test iterations')
    parser.add_argument('--num_warmup', default=-1, type=int, help='test warmup')
    parser.add_argument('--dataset_dir', default='data/METR-LA', type=str, help='data location')
    parser.add_argument("--compile", action='store_true', default=False,
                    help="enable torch.compile")
    parser.add_argument("--backend", type=str, default='inductor',
                    help="enable torch.compile backend")

    args = parser.parse_args()
    run_dcrnn(args)
