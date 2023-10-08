# A Basic ML Workflow That Runs On Orfeo
This repo showcases a minimal but complete working example of an ML 
project that can utilize pretty much any multi-gpu system including the 
GPU nodes on [Orfeo](https://orfeo-doc.areasciencepark.it/).

The workflow consists of downloading the 
[IMDB review sentiment dataset](https://huggingface.co/datasets/imdb)
from Hugging Face Datasets and then fine-tuning a 
[pre-trained BERT model](https://huggingface.co/bert-large-uncased) on 
it.
Training on multiple GPUs (and even multiple nodes) is made easy 
through the use of the 
[Accelerate](https://slurm.schedmd.com/sbatch.html) package.
The goal of this repo is to provide a basic starting point for any ML 
project that wishes to utilize the resources present on Orfeo.

## Config
Unfortunately, there are 3 places where configuration needs to be done.
An Accelerate config file needs to be generated, your Slurm sbatch 
parameters need to be defined in job_script.sh, and the model params 
need to be defined either in a .env file or in the environment.

Ideally I'd love to centralize everything in a .env file, but that's 
impossible as neither Slurm nor Accelerate have proper support for using 
env variables for configuration.

The accelerate config file needs to be generated in the project root 
with the name "acclerate_config.yaml". This can be done by running (from 
the project root directory on any computer):
```shell
accelerate config --config_file ./accelerate_config.yaml
```
This will ask you a series of questions about how you'd like to run your 
model train script. You can decide here how many GPUs and nodes you'd 
like to use.
If you want to store your config file somewhere else or want it to have 
a different name, you can set the "MLP_ACCELERATE_CONFIG" env variable 
to the location of the file. More documentation on Accelerate can be 
found on their [website](https://huggingface.co/docs/accelerate/index).

You also have to set your Slurm sbatch parameters in job_script.sh.
Documentation on these can be found on the Slurm 
[website](https://slurm.schedmd.com/sbatch.html).

Lastly, the model script itself expects certain environment variables to 
be set. All possible values can be found in /src/settings.py, and can be 
set as environment variables or in a .env file in the project root.
All env variables must have the prefix "MLP_" for example, to set the 
training output directory, one can do:
```shell
export MLP_OUTPUT_DIR=./train
```

## File Sync With Orfeo
I have had the best experience using SFTP to synchronize my project 
directory with some directory in my home drive in Orfeo.
My main IDE PyCharm can automatically do this, but there are many tools, 
such as rsync for example, that can do it as well.

When doing file sync, there are some things to be careful off. 
Make sure to avoid syncing unnecessary files that may have sensitive 
info such as the .git or .idea files. Also avoid syncing any local venv 
directory that you may be using to test your code.
Lastly, try to avoid syncing large files. These should only be 
transferred once to or from Orfeo and never transferred again.
This would include the training output directory and potentially a 
dataset directory if your dataset is large.
