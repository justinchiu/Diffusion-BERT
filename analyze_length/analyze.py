import pathlib
import math
import torch
import pandas as pd

DIR = pathlib.Path("model_bert-base-uncased_bsz_32_lr_5e-05_seed_42_numsteps_2048_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.3_fromscratch_False_timestep_layerwise_ckpts/elbos")
SCRATCHDIR = pathlib.Path("model_bert-base-uncased_bsz_32_lr_5e-05_seed_42_numsteps_2048_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.3_fromscratch_True_timestep_layerwise_ckpts/elbos")
LARGEDIR = pathlib.Path("model_bert-large-uncased_bsz_32_lr_5e-05_seed_42_numsteps_2048_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.3_fromscratch_False_timestep_layerwise_ckpts/elbos")

step_sizes = [4, 16, 32, 128, 256, 1024]

start_end_pairs = [
    (0, 32),
    (32, 64),
    (64, 96),
    #(96, 128),
]

steps = []
lengths = []
ppls = []
for model in ["base-pretrained", "base-scratch", "large-pretrained"]:
for size in step_sizes:
    for s,t in start_end_pairs:
        xs = torch.load(DIR / f"elbo-avg-by-lens-chp-104999-totsteps-2048-stepsize-{size}-minlen-{s}-maxlen-{t}-nb-10.th", map_location="cpu")
        xs = torch.load(SCRATCHDIR / f"simple-elbo-avg-by-lens-chp-104999-totsteps-2048-stepsize-{size}-minlen-{s}-maxlen-{t}-nb-10.th", map_location="cpu")
        xs = torch.load(LARGEDIR / f"simple-elbo-avg-by-lens-chp-194999-totsteps-2048-stepsize-{size}-minlen-{s}-maxlen-{t}-nb-10.th", map_location="cpu")

        elbo = xs["elbo"]
        avg_elbo = xs["avg_token_elbo"]
        avg_ppl = math.exp(avg_elbo)
        num_tokens = xs["num_tokens"]
        #num_examples = xs["num_examples"]
        steps.append(2048//size)
        lengths.append((s,t))
        ppls.append(avg_ppl)

df = pd.DataFrame({
    "steps": steps,
    "lengths": lengths,
    "ppls": ppls,
})

print(df)
import pdb; pdb.set_trace()
