import pathlib
import math
import torch
import pandas as pd

DIR = pathlib.Path("model_bert-base-uncased_bsz_32_lr_5e-05_seed_42_numsteps_2048_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.3_fromscratch_False_timestep_layerwise_ckpts/elbos")
SCRATCHDIR = pathlib.Path("model_bert-base-uncased_bsz_32_lr_5e-05_seed_42_numsteps_2048_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.3_fromscratch_True_timestep_layerwise_ckpts/elbos")
LARGEDIR = pathlib.Path("model_bert-large-uncased_bsz_32_lr_5e-05_seed_42_numsteps_2048_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.3_fromscratch_False_timestep_layerwise_ckpts/elbos")
BIGSDIR = pathlib.Path("model_JunxiongWang/BiGS_512_bsz_32_lr_5e-05_seed_42_numsteps_2048_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.3_fromscratch_False_timestep_layerwise_ckpts/elbos")
BIGSSCRATCHDIR = pathlib.Path("model_JunxiongWang/BiGS_512_bsz_32_lr_5e-05_seed_42_numsteps_2048_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.3_fromscratch_True_timestep_layerwise_ckpts/elbos")

step_sizes = [4, 16, 32, 128, 256, 1024]

start_end_pairs = [
    (0, 32),
    (32, 64),
    (64, 96),
    (96, 128),
]

models = []
steps = []
lengths = []
ppls = []
num_xs = []
for model in ["bert-base", "bert-base-scratch", "bert-large", "bigs-large", "bigs-large-scratch"]:
    for size in step_sizes:
        for s,t in start_end_pairs:
            if model == "bert-base":
                xs = torch.load(DIR / f"simple-elbo-avg-by-lens-chp-104999-totsteps-2048-stepsize-{size}-minlen-{s}-maxlen-{t}-nb-10.th", map_location="cpu")
            elif model == "bert-base-scratch":
                xs = torch.load(SCRATCHDIR / f"simple-elbo-avg-by-lens-chp-104999-totsteps-2048-stepsize-{size}-minlen-{s}-maxlen-{t}-nb-10.th", map_location="cpu")
            elif model == "bert-large":
                xs = torch.load(LARGEDIR / f"simple-elbo-avg-by-lens-chp-194999-totsteps-2048-stepsize-{size}-minlen-{s}-maxlen-{t}-nb-10.th", map_location="cpu")
            elif model == "bigs-large":
                xs = torch.load(BIGSDIR / f"simple-elbo-avg-by-lens-chp-209999-totsteps-2048-stepsize-{size}-minlen-{s}-maxlen-{t}-nb-10.th", map_location="cpu")
            elif model == "bigs-large-scratch":
                xs = torch.load(BIGSSCRATCHDIR / f"simple-elbo-avg-by-lens-chp-209999-totsteps-2048-stepsize-{size}-minlen-{s}-maxlen-{t}-nb-10.th", map_location="cpu")

            elbo = xs["elbo"]
            avg_elbo = xs["avg_token_elbo"]
            avg_ppl = math.exp(avg_elbo)
            num_tokens = xs["num_tokens"]
            num_examples = xs["num_examples"]
            models.append(model)
            steps.append(2048//size)
            lengths.append((s,t))
            ppls.append(avg_ppl)
            num_xs.append(num_examples)
            import pdb; pdb.set_trace()

df = pd.DataFrame({
    "model": models,
    "steps": steps,
    "lengths": lengths,
    "ppls": ppls,
    "n": num_xs,
})
print(df)
df.to_csv("analyze_length/results.csv")
import pdb; pdb.set_trace()
