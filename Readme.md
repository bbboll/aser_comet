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