import pathlib
import torch

DIR = pathlib.Path("model_bert-base-uncased_bsz_32_lr_5e-05_seed_42_numsteps_2048_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.3_fromscratch_False_timestep_layerwise_ckpts/elbos")
#DIR = pathlib.Path("model_bert-large-uncased_bsz_32_lr_5e-05_seed_42_numsteps_2048_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.3_fromscratch_False_timestep_layerwise_ckpts/elbos")

step_sizes = [4, 16, 32, 128, 256, 1024]

start_end_pairs = [
    (0, 32),
    (32, 64),
    (64, 96),
    (96, 128),
]

for size in step_sizes:
    for s,t in start_end_pairs:
        xs = torch.load(DIR / f"elbo-avg-by-lens-chp-74999-totsteps-2048-stepsize-{size}-minlen-{s}-maxlen-{t}-nb-10.th", map_location="cpu")
        #xs = torch.load(DIR / f"elbo-avg-by-lens-chp-194999-totsteps-2048-stepsize-{size}-minlen-{s}-maxlen-{t}-nb-10.th", map_location="cpu")
        import pdb; pdb.set_trace()
