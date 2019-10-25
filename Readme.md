# Installation
Follow instructions in the [COMET submodule](https://github.com/atcbosselut/comet-commonsense) to download model files and install dependencies.

# Resources

COMET
https://github.com/atcbosselut/comet-commonsense

ASER API
https://github.com/HKUST-KnowComp/ASER

GPT
https://github.com/openai/finetune-transformer-lm

GPT-2
https://openai.com/blog/better-language-models/
https://github.com/openai/gpt-2

Clone including submodules:
git clone --recurse-submodules -j8 https://github.com/bbboll/aser_comet

# Notes

Seems like the COMET code is not compatible with modern versions of tensorboardX. As a workaround, one can comment out the last line in
`COMET/src/train/train.py: Trainer.set_logger(self)`

Training results saved in
models/aser-generation/iteration-50000-1000000/transformer/maxe1_18-maxe2_20-maxr_1/model_transformer-nL_12-nH_12-hSize_768-edpt_0.1-adpt_0.1-rdpt_0.1-odpt_0.1-pt_gpt-afn_gelu-init_pt-vSize_40519/exp_generation-seed_123-l2_0.01-vl2_T-lrsched_warmup_linear-lrwarm_0.002-clip_1-loss_nll-b2_0.999-b1_0.9-e_1e-08/bs_1-smax_40-sample_greedy-numseq_1-gs_1000-es_1000/