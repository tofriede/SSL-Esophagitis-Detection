#!/usr/bin/env python3
import argparse
import json
import logging
import os
import time
import csv
from contextlib import suppress
from functools import partial
from sys import maxsize
from collections import OrderedDict
from glob import glob
from typing import List, Optional, Any

import yaml
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, confusion_matrix
from dotenv import load_dotenv
from timm.data import create_dataset, create_loader, resolve_data_config, ImageNetInfo, infer_imagenet_subset
from timm.layers import apply_test_time_pool
from timm.models import create_model
from timm.utils import AverageMeter, setup_default_logging, set_jit_fuser, ParseKwargs

from esodetector.utils.eval import get_class1_metrics, save_confusion_matrix_heatmap, calculate_auc_roc_scores, calculate_auc_pr_scores

load_dotenv()

try:
    from apex import amp
    has_apex = True
except ImportError:
    has_apex = False

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')


_FMT_EXT = {
    'json': '.json',
    'json-record': '.json',
    'json-split': '.json',
    'parquet': '.parquet',
    'csv': '.csv',
}

LABELS = OrderedDict({
    0: 'other',
    1: 'esophagitis',
})

DATA_DIR = os.getenv('DATA_DIR')
CLASS_MAP = os.getenv('CLASS_MAP')

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('inference')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Validation Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch Esophagitis Inference')
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (*deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR', default=DATA_DIR,
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--model', '-m', metavar='MODEL', default='resnet50',
                    help='model architecture (default: resnet50)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--in-chans', type=int, default=None, metavar='N',
                    help='Image input channels (default: None => 3)')
parser.add_argument('--input-size', default=None, nargs=3, type=int, metavar='N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--use-train-size', action='store_true', default=False,
                    help='force use of train input size, even when test size is specified in pretrained cfg')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--crop-mode', default=None, type=str,
                    metavar='N', help='Input image crop mode (squash, border, center). Model default if None.')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default=CLASS_MAP, type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--multi-seed', default=False,
                    help='evaluate multiple models trained with different seeds')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")
parser.add_argument('--amp', action='store_true', default=False,
                    help='use Native AMP for mixed precision training')
parser.add_argument('--amp-dtype', default='float16', type=str,
                    help='lower precision AMP dtype (default: float16)')
parser.add_argument('--model-dtype', default=None, type=str,
                   help='Model dtype override (non-AMP) (default: float32)')
parser.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
parser.add_argument('--model-kwargs', nargs='*', default={}, action=ParseKwargs)
parser.add_argument('--torchcompile-mode', type=str, default=None,
                    help="torch.compile mode (default: None).")

scripting_group = parser.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', default=False, action='store_true',
                             help='torch.jit.script the full model')
scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")
scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
                             help="Enable AOT Autograd support.")

parser.add_argument('--results-dir', type=str, default=None,
                    help='folder for output results')
parser.add_argument('--results-file', type=str, default=None,
                    help='results filename (relative to results-dir)')
parser.add_argument('--results-format', type=str, nargs='+', default=['csv'],
                    help='results format (one of "csv", "json", "json-split", "parquet")')
parser.add_argument('--results-separate-col', action='store_true', default=False,
                    help='separate output columns per result index.')
parser.add_argument('--topk', default=1, type=int,
                    metavar='N', help='Top-k to output to CSV')
parser.add_argument('--fullname', action='store_true', default=False,
                    help='use full sample name in output (not just basename).')
parser.add_argument('--filename-col', type=str, default='filename',
                    help='name for filename / sample name column')
parser.add_argument('--index-col', type=str, default='index',
                    help='name for output indices column(s)')
parser.add_argument('--label-col', type=str, default='label',
                    help='name for output indices column(s)')
parser.add_argument('--output-col', type=str, default=None,
                    help='name for logit/probs output column(s)')
parser.add_argument('--output-type', type=str, default='prob',
                    help='output type colum ("prob" for probabilities, "logit" for raw logits)')
parser.add_argument('--label-type', type=str, default='description',
                    help='type of label to output, one of  "none", "name", "description", "detailed"')
parser.add_argument('--include-index', action='store_true', default=False,
                    help='include the class index in results')
parser.add_argument('--exclude-output', action='store_true', default=False,
                    help='exclude logits/probs from results, just indices. topk must be set !=0.')
parser.add_argument('--no-console-results', action='store_true', default=False,
                    help='disable printing the inference results to the console')
parser.add_argument('--metrics-avg', type=str, default='macro',
                    choices=['micro', 'macro', 'weighted'],
                    help='Enable precision, recall, F1-score calculation and specify the averaging method.')


def _parse_args() -> argparse.Namespace:
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    return args


def find_checkpoint_files(checkpoint_path: str) -> List[str]:
    """Find all model_best.pth.tar files recursively in the checkpoint directory."""
    checkpoint_files = []
    for filepath in glob(os.path.join(checkpoint_path, '**', 'model_best.pth.tar'), recursive=True):
        checkpoint_files.append(filepath)
    return sorted(checkpoint_files)


def validate_single_checkpoint(
    args: argparse.Namespace,
    checkpoint_path: str,
    results_subdir: Optional[str] = None
) -> OrderedDict:
    """Run validation for a single checkpoint."""
    # Update checkpoint path
    args.checkpoint = checkpoint_path
    
    # Update results directory if subdir is specified
    original_results_dir = args.results_dir
    if results_subdir and args.results_dir:
        args.results_dir = os.path.join(args.results_dir, results_subdir)
    
    _logger.info(f'Validating checkpoint: {checkpoint_path}')
    if args.results_dir:
        _logger.info(f'Results will be saved to: {args.results_dir}')

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    model_dtype = None
    if args.model_dtype:
        assert args.model_dtype in ('float32', 'float16', 'bfloat16')
        model_dtype = getattr(torch, args.model_dtype)

    # resolve AMP arguments based on PyTorch / Apex availability
    amp_autocast = suppress
    if args.amp:
        assert model_dtype is None or model_dtype == torch.float32, 'float32 model dtype must be used with AMP'
        assert args.amp_dtype in ('float16', 'bfloat16')
        amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        _logger.info('Running inference in mixed precision with native PyTorch AMP.')
    else:
        _logger.info('Running inference in float32. AMP not enabled.')

    if args.fuser:
        set_jit_fuser(args.fuser)

    # create model
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=in_chans,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint,
        **args.model_kwargs,
    )
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    _logger.info(
        f'Model {args.model} created, param count: {sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model)
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config)

    model = model.to(device=device, dtype=model_dtype)
    model.eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.torchscript:
        model = torch.jit.script(model)
    elif args.torchcompile:
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
        torch._dynamo.reset()
        model = torch.compile(model, backend=args.torchcompile, mode=args.torchcompile_mode)
    elif args.aot_autograd:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    if args.class_map and isinstance(args.class_map, str) and os.path.splitext(args.class_map)[-1].lower() == '.json':
        with open(args.class_map, 'r') as f:
            args.class_map = json.load(f)

    root_dir = args.data or args.data_dir
    dataset = create_dataset(
        root=root_dir,
        name=args.dataset,
        split=args.split,
        class_map=args.class_map,
    )

    if test_time_pool:
        data_config['crop_pct'] = 1.0

    workers = 1 if 'tfds' in args.dataset or 'wds' in args.dataset else args.workers
    loader = create_loader(
        dataset,
        batch_size=args.batch_size,
        use_prefetcher=True,
        num_workers=workers,
        device=device,
        img_dtype=model_dtype or torch.float32,
        **data_config,
    )

    to_label = None
    if args.label_type in ('name', 'description', 'detail'):
        imagenet_subset = infer_imagenet_subset(model)
        if imagenet_subset is not None:
            dataset_info = ImageNetInfo(imagenet_subset)
            if args.label_type == 'name':
                to_label = lambda x: dataset_info.index_to_label_name(x)
            elif args.label_type == 'detail':
                to_label = lambda x: dataset_info.index_to_description(x, detailed=True)
            else:
                to_label = lambda x: dataset_info.index_to_description(x)
        else:
            to_label = lambda x: LABELS.get(x, str(x))
    to_label = np.vectorize(to_label)

    top_k = min(args.topk, args.num_classes)
    batch_time = AverageMeter()
    end = time.time()
    all_indices = []
    all_labels = []
    all_outputs = []
    all_preds = []
    all_targets = []
    all_probabilities = []  # For AUC calculation
    use_probs = args.output_type == 'prob'
    with torch.inference_mode():
        for batch_idx, (input, target) in enumerate(loader):

            with amp_autocast():
                output = model(input)

            predictions = torch.argmax(output, dim=1)
            all_preds.append(predictions.cpu())
            all_targets.append(target.cpu())
            
            # Always compute probabilities for AUC calculation
            probabilities = output.softmax(-1)
            all_probabilities.extend(probabilities.float().cpu().numpy())

            if use_probs:
                output = output.softmax(-1)

            if top_k:
                output, indices = output.topk(top_k)
                np_indices = indices.cpu().numpy()
                if args.include_index:
                    all_indices.append(np_indices)
                if to_label is not None:
                    np_labels = to_label(np_indices)
                    all_labels.append(np_labels)

            all_outputs.append(output.float().cpu().numpy())
         

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                _logger.info('Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    batch_idx, len(loader), batch_time=batch_time))

    all_indices = np.concatenate(all_indices, axis=0) if all_indices else None
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else None
    all_outputs = np.concatenate(all_outputs, axis=0).astype(np.float32)
    filenames = loader.dataset.filenames(basename=not args.fullname)

    output_col = args.output_col or ('prob' if use_probs else 'logit')
    data_dict = {args.filename_col: filenames}
    if args.results_separate_col and all_outputs.shape[-1] > 1:
        if all_indices is not None:
            for i in range(all_indices.shape[-1]):
                data_dict[f'{args.index_col}_{i}'] = all_indices[:, i]
        if all_labels is not None:
            for i in range(all_labels.shape[-1]):
                data_dict[f'{args.label_col}_{i}'] = all_labels[:, i]
        for i in range(all_outputs.shape[-1]):
            data_dict[f'{output_col}_{i}'] = all_outputs[:, i]
    else:
        if all_indices is not None:
            if all_indices.shape[-1] == 1:
                all_indices = all_indices.squeeze(-1)
            data_dict[args.index_col] = list(all_indices)
        if all_labels is not None:
            if all_labels.shape[-1] == 1:
                all_labels = all_labels.squeeze(-1)
            data_dict[args.label_col] = list(all_labels)
        if all_outputs.shape[-1] == 1:
            all_outputs = all_outputs.squeeze(-1)
        data_dict[output_col] = list(all_outputs)

    df = pd.DataFrame(data=data_dict)

    metric_results = {}
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_probabilities = np.array(all_probabilities)
    
    f1 = f1_score(all_targets, all_preds, average=args.metrics_avg, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    sensitivity_class1, specificity_class1, precision_class1 = get_class1_metrics(cm, all_targets)
    
    # Calculate AUC scores
    auc_roc_scores = calculate_auc_roc_scores(all_targets, all_probabilities, args.num_classes)
    auc_roc = auc_roc_scores['auc']

    auc_pr_scores = calculate_auc_pr_scores(all_targets, all_probabilities, args.num_classes)
    auc_pr = auc_pr_scores['auc']
    
    metric_results = {
        'auc_pr': round(100 * auc_pr, 4),
        'sensitivity_class1': round(100 * sensitivity_class1, 4),
        'specificity_class1': round(100 * specificity_class1, 4),
        'precision_class1': round(100 * precision_class1, 4),
        f'{args.metrics_avg}_f1_score': round(100 * f1, 4),
        'auc_roc': round(100 * auc_roc, 4),
    }

    results = OrderedDict(
        **metric_results,
        img_size=data_config['input_size'][-1],
        interpolation=data_config['interpolation'],
    )

    results_filename = args.results_file
    if results_filename:
        filename_no_ext, ext = os.path.splitext(results_filename)
        if ext and ext in _FMT_EXT.values():
            # if filename provided with one of expected ext,
            # remove it as it will be added back
            results_filename = filename_no_ext
    else:
        # base default filename on model name + img-size
        img_size = data_config["input_size"][1]
        results_filename = f'{args.model}-{img_size}'

    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        metrics_filename = os.path.join(args.results_dir, f'metrics_{results_filename}.json')
        confusion_matrix_filename = os.path.join(args.results_dir, f'confusion_matrix_{results_filename}.png')
        results_filename = os.path.join(args.results_dir, f'samples_{results_filename}')
    else:
        metrics_filename = f'metrics_{results_filename}.json'
        confusion_matrix_filename = f'confusion_matrix_{results_filename}.png'
        results_filename = f'samples_{results_filename}'

    for fmt in args.results_format:
        save_results(df, results_filename, fmt)

    write_metric_results(metrics_filename, results, format='json')
    save_confusion_matrix_heatmap(cm, list(LABELS.values()), confusion_matrix_filename)

    if not args.no_console_results:
        print(f'--result')
        print(df.set_index(args.filename_col).to_json(orient='index', indent=4))
    
    # Restore original results directory
    args.results_dir = original_results_dir
    
    return results


def main() -> None:
    setup_default_logging()
    args = _parse_args()

    # might as well try to do something useful...
    args.pretrained = args.pretrained or not args.checkpoint

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    # Handle multi-seed validation
    if args.multi_seed and args.checkpoint:
        # Check if checkpoint is a directory
        if os.path.isdir(args.checkpoint) and not args.checkpoint.endswith('.pth.tar'):
            _logger.info(f'Multi-seed mode enabled. Searching for model_best.pth.tar files in: {args.checkpoint}')
            checkpoint_files = find_checkpoint_files(args.checkpoint)
            
            if not checkpoint_files:
                _logger.error(f'No model_best.pth.tar files found in {args.checkpoint}')
                return
            
            _logger.info(f'Found {len(checkpoint_files)} checkpoint(s) to validate')
            
            for checkpoint_file in checkpoint_files:
                parent_dir = os.path.basename(os.path.dirname(checkpoint_file))
                
                _logger.info(f'\n{"="*80}')
                _logger.info(f'Processing checkpoint: {checkpoint_file}')
                _logger.info(f'Results subdirectory: {parent_dir}')
                _logger.info(f'{"="*80}\n')
                
                # Run validation for this checkpoint
                validate_single_checkpoint(args, checkpoint_file, results_subdir=parent_dir)
            
            return
    
    # Single checkpoint validation (original behavior)
    validate_single_checkpoint(args, args.checkpoint)


def save_results(
    df: pd.DataFrame,
    results_filename: str,
    results_format: str = 'csv',
    filename_col: str = 'filename'
) -> None:
    np.set_printoptions(threshold=maxsize)
    results_filename += _FMT_EXT[results_format]
    if results_format == 'parquet':
        df.set_index(filename_col).to_parquet(results_filename)
    elif results_format == 'json':
        df.set_index(filename_col).to_json(results_filename, indent=4, orient='index')
    elif results_format == 'json-records':
        df.to_json(results_filename, lines=True, orient='records')
    elif results_format == 'json-split':
        df.to_json(results_filename, indent=4, orient='split', index=False)
    else:
        df.to_csv(results_filename, index=False)


def write_metric_results(
    results_file: str,
    results: Any,
    format: str = 'csv'
) -> None:
    with open(results_file, mode='w') as cf:
        if format == 'json':
            json.dump(results, cf, indent=4)
        else:
            if not isinstance(results, (list, tuple)):
                results = [results]
            if not results:
                return
            dw = csv.DictWriter(cf, fieldnames=results[0].keys())
            dw.writeheader()
            for r in results:
                dw.writerow(r)
            cf.flush()


if __name__ == '__main__':
    main()