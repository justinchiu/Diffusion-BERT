"""
Evaluate the ELBO of diffusionbert on LM1B test
"""
import os
import sys
import random
import numpy as np
import argparse
import torch
from dataloader import DiffusionLoader
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaConfig
from models.modeling_roberta import RobertaForMaskedLM
import diffusion_word_freq
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sample import Categorical, WholeWordMasking
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import datetime

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str, required=False)
    parser.add_argument("--task_name", default='lm1b', type=str, required=False)
    parser.add_argument("--lr", default=5e-4, type=float, required=False)
    parser.add_argument("--epochs", default=3, type=int, required=False)
    parser.add_argument("--batch_size", default=64, type=int, required=False)
    parser.add_argument("--word_freq_lambda", default=0.3, type=float, required=False)
    parser.add_argument("--num_steps", default=2048, type=int, required=False)
    parser.add_argument("--eval_step_size", default=4, type=int, required=False)
    parser.add_argument("--dev_size", default=5e-4, type=float, required=False)
    parser.add_argument("--hybrid_lambda", default=1e-2, type=float, required=False)
    parser.add_argument("--eval_steps", default=15000, type=int, required=False)
    parser.add_argument("--seed", default=42, type=int, required=False)
    # parser.add_argument("--device", default='cuda:0', type=str, required=False)
    parser.add_argument("--logging_steps", default=1000, type=int, required=False)
    parser.add_argument('--predict_x0', default=True, type=bool, required=False)
    parser.add_argument("--load_step", default=-1, type=int, required=False)
    parser.add_argument("--sample_strategy", default='Categorical', type=str, required=False)
    parser.add_argument("--schedule", default='mutual', type=str, required=False)
    parser.add_argument("--from_scratch", default=False, type=bool, required=False)
    parser.add_argument("--timestep", default='none', type=str, required=False)
    parser.add_argument("--length_min", default=0, type=int, required=True)
    parser.add_argument("--length_max", default=32, type=int, required=True)
    parser.add_argument("--num_batches", default=10, type=int, required=False)
    return parser.parse_args()


if __name__ == '__main__':
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_P2P_LEVEL'] = 'NVL'
    args = parse_args()
    print(args)
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device("cuda", local_rank)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        timeout=datetime.timedelta(seconds=9600)
    )

    set_seed(args)
    if args.timestep in ['none', 'token']:
        from models.modeling_bert import BertForMaskedLM
    elif args.timestep == 'layerwise':
        from models.modeling_bert_new_timestep import BertForMaskedLM
    else:
        raise NotImplementedError


    save_path = f'./model_{args.model_name_or_path}_bsz_{args.batch_size}_lr_{args.lr}_seed_{args.seed}_numsteps_{args.num_steps}_sample_{args.sample_strategy}_schedule_{args.schedule}_hybridlambda_{args.hybrid_lambda}_wordfreqlambda_{args.word_freq_lambda}_fromscratch_{args.from_scratch}_timestep_{args.timestep}_ckpts'

    bigs_models = [
        "JunxiongWang/BiGS_128",
        "JunxiongWang/BiGS_512",
    ]
    if args.model_name_or_path in ['bert-base-uncased', 'bert-large-uncased']:
        model_cls = BertForMaskedLM
        cfg_cls = BertConfig
        tok_cls = BertTokenizer
    elif args.model_name_or_path in ['roberta-base']:
        model_cls = RobertaForMaskedLM
        cfg_cls = RobertaConfig
        tok_cls = RobertaTokenizer
    elif args.model_name_or_path in bigs_models:
        from BiGS.modeling_bigs import BiGSForMaskedLM, BiGSConfig
        model_cls = BiGSForMaskedLM
        cfg_cls = BiGSConfig
        tok_cls = BertTokenizer
    else:
        raise NotImplementedError


    tokenizer = tok_cls.from_pretrained(args.model_name_or_path)
    word_freq = torch.load(
        f'./word_freq/{args.model_name_or_path}_{args.task_name}.pt'
        if args.model_name_or_path not in bigs_models + ["bert-large-uncased"]
        else f'./word_freq/bert-base-uncased_{args.task_name}.pt'
    )
    assert word_freq.size(0) == tokenizer.vocab_size


    def word_freq_preprocess_fn(wf):
        wf = wf + 1
        wf = wf.log()
        wf = wf / wf.max()

        # range: 0 - 1
        return wf

    def process_fn_in_collate(wf):
        return wf - wf.mean()

    word_freq = word_freq_preprocess_fn(word_freq)

    word_freq[tokenizer.pad_token_id] = 0.  # stable training

    if args.sample_strategy == 'Categorical':
        sample_cls = Categorical()
    elif args.sample_strategy == 'wwm':
        sample_cls = WholeWordMasking(tokenizer)
    else:
        raise ValueError

    diffusion_schedule = diffusion_word_freq.create_discrete_diffusion_schedule(args.schedule, num_steps=args.num_steps)
    diffusion_instance = diffusion_word_freq.MaskDiffusion(
        dim=tokenizer.vocab_size,
        schedule=diffusion_schedule,
        tokenizer=tokenizer,
        sample_cls=sample_cls,
        word_freq_lambda=args.word_freq_lambda,
        device=device
    )

    # must load for evaluation!
    assert args.load_step > 0
    if args.load_step > 0:
        path = os.path.join(save_path, f'best({args.load_step}).th')
        print(f"loading {path}")
        ckpt = torch.load(path)
    cfg = cfg_cls.from_pretrained(args.model_name_or_path)
    cfg.overall_timestep = diffusion_instance.num_steps

    # must load for evaluation!
    assert args.load_step >= 0
    model = model_cls(cfg).to(device)

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in ckpt["model"].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    #model.load_state_dict(ckpt['model'])

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    train_data, test_data = DiffusionLoader(tokenizer=tokenizer).my_load(task_name='lm1b', splits=['train', 'test'])
    train_data, dev_data = train_data.train_test_split(test_size=args.dev_size).values()

    if dist.get_rank() == 0:
        print('# of train data: {}'.format(len(train_data)))
        print('Example:')
        print(train_data[0])
        print('\n# of dev data: {}'.format(len(dev_data)))
        print('Example:')
        print(dev_data[0])
        print('\n# of test data: {}'.format(len(test_data)))
        print('Example:')
        print(test_data[0])

    def collate_fn(batch_input):
        input_ids = [torch.tensor(d['input_ids']) for d in batch_input]
        attention_mask = [torch.tensor(d['attention_mask']) for d in batch_input]
        word_freq_logits = [process_fn_in_collate(word_freq.gather(0, torch.tensor(d['input_ids']))) for d in batch_input]
        input_ids = pad_sequence(input_ids, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        word_freq_logits = pad_sequence(word_freq_logits, batch_first=True)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'word_freq_logits': word_freq_logits
        }


    # FILTER TEST DATA TO CORRECT LENGTHS
    test_data = test_data.filter(lambda example:
        args.length_min < sum(example["attention_mask"])
        and sum(example["attention_mask"]) <= args.length_max
    )

    dev_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    dev_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size * 2, collate_fn=collate_fn, num_workers=4, pin_memory=True, sampler=dev_sampler)

    cls = torch.full((1, 1), fill_value=tokenizer.cls_token_id, device=device)
    sep = torch.full((1, 1), fill_value=tokenizer.sep_token_id, device=device)

    att_ones = torch.ones((1, 1), device=device)
    att_zeros = torch.zeros((1, 1), device=device)

    if args.timestep == 'none':
        def denoise_fn(targets, timestep, attention_mask):
            assert len(targets.size()) == 2  # bsz * seqlen
            bsz = targets.size(0)
            targets = torch.cat((cls.repeat(bsz, 1), targets, sep.repeat(bsz, 1)), dim=1)
            attention_mask = torch.cat((att_ones.repeat(bsz, 1), attention_mask, att_zeros.repeat(bsz, 1)), dim=1)
            return model(input_ids=targets, timestep=timestep - 1, attention_mask=attention_mask)['logits'][:, 1:-1, :]
    elif args.timestep == 'token':

        def denoise_fn(targets, timestep, attention_mask):
            assert len(targets.size()) == 2  # bsz * seqlen
            bsz = targets.size(0)
            targets = torch.cat((
                cls.repeat(bsz, 1),
                torch.full((bsz, 1), fill_value=timestep.item() + 110, device=device),
                targets,
                sep.repeat(bsz, 1)
            ), dim=1)
            attention_mask = torch.cat((att_ones.repeat(bsz, 2), attention_mask, att_zeros.repeat(bsz, 1)), dim=1)
            return model(input_ids=targets, timestep=timestep - 1, attention_mask=attention_mask)['logits'][:, 2:-1, :]
    elif args.timestep == 'layerwise':
        def denoise_fn(targets, timestep, attention_mask):
            assert len(targets.size()) == 2  # bsz * seqlen
            bsz = targets.size(0)
            targets = torch.cat((
                cls.repeat(bsz, 1),
                targets,
                sep.repeat(bsz, 1)
            ), dim=1)
            attention_mask = torch.cat((att_ones.repeat(bsz, 1), attention_mask, att_zeros.repeat(bsz, 1)), dim=1)
            return model(input_ids=targets, timestep=timestep - 1, attention_mask=attention_mask)['logits'][:, 1:-1, :]
    else:
        raise NotImplementedError

    nan_count = 0
    loss_list = [torch.tensor(0., device=device) for _ in range(dist.get_world_size())]
    nan_count_in_dev = 0
    model.eval()

    MAX_LEN = 256
    BSZ = args.batch_size * 2

    dev_metrics = {
        'elbo': torch.zeros(MAX_LEN, dtype=torch.float32, device=device),
        'elbo_in_bits_per_dim': torch.zeros(MAX_LEN, dtype=torch.float32, device=device),
        # 'likelihood': .0,
        # 'prior': .0,
    }
    length_counts = torch.zeros(MAX_LEN, dtype=torch.int64, device=device)

    with torch.no_grad():
        batch_ones = torch.ones(args.batch_size*2, dtype=torch.int64, device=device)

        num_batches = 0
        for dev_batch in tqdm(dev_loader):
            batch_dev_metrics = diffusion_word_freq.discrete_diffusion_elbo(
                dev_batch['input_ids'].to(device),
                denoise_fn=denoise_fn,
                diffusion=diffusion_instance,
                target_mask=dev_batch['attention_mask'].to(device),
                normalize_without_padding=True,
                eval_step_size=args.eval_step_size,
                word_freq_logits=dev_batch['word_freq_logits'].to(device),
                device=device,
                per_example=True,
            )
            # count_nonzero should have a device kwarg, but doesnt
            lengths = torch.count_nonzero(dev_batch["attention_mask"], axis=-1).to(device)

            batch_metrics_by_length = {
                name: torch.zeros(MAX_LEN, dtype=torch.float32, device=device)
                    .scatter_add_(0, lengths, batch_dev_metrics[name])
                for name in dev_metrics.keys()
            }
            batch_length_counts = (torch.zeros(MAX_LEN, dtype=torch.int64, device=device)
                .scatter_add_(0, lengths, batch_ones)
            )

            if dist.get_rank() == 0:
                m = [torch.zeros(MAX_LEN, device=device) for _ in range(dist.get_world_size())]
                for name in dev_metrics.keys():
                    dist.gather(batch_metrics_by_length[name].squeeze(), m)
                    temp = sum(m)
                    if not torch.isnan(temp).any():
                        dev_metrics[name] += temp
                    else:
                        nan_count_in_dev += 1
                        logger.warning(f'NaN encountered {nan_count_in_dev} times in dev')

                length_m = [torch.zeros(MAX_LEN, dtype=torch.int64, device=device) for _ in range(dist.get_world_size())]
                dist.gather(batch_length_counts, length_m)

                length_counts += sum(length_m)
            else:
                for name in dev_metrics.keys():
                    dist.gather(batch_metrics_by_length[name].squeeze())
                    dist.gather(batch_length_counts)

            num_batches += 1
            if num_batches >= args.num_batches:
                break

        if dist.get_rank() == 0:
            elbo = dev_metrics["elbo"]
            avg_token_elbo = elbo / (length_counts * torch.arange(MAX_LEN, device=device))
            os.makedirs(f"{save_path}/elbos", exist_ok=True)
            elbo_save_path = f'{save_path}/elbos/elbo-avg-by-lens-chp-{args.load_step}-totsteps-{args.num_steps}-stepsize-{args.eval_step_size}-minlen-{args.length_min}-maxlen-{args.length_max}-nb-{args.num_batches}.th'
            print(f"SAVING TO {elbo_save_path}")
            torch.save({
                "elbo": elbo,
                "avg_token_elbo": avg_token_elbo,
            }, elbo_save_path)
