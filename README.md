# sahajBERT

## Downstream evaluation

We have two downstream task `NER` and `NCC`

The datasets have been used here are:

NER: `wikiann`, `bn`

NCC: `indic_glue`, `sna.bn`

To read more about the datasets visit [WikiANN](https://huggingface.co/datasets/wikiann), [IndicGLUE](https://huggingface.co/datasets/indic_glue)

Model link - [sahajBERT-xlarge](https://huggingface.co/Upload/sahajbert2)

### NER

##### 1. Clone the sahajbert repo and prepare the env by intalling requirements.
```
git clone https://github.com/tanmoyio/sahajbert.git
cd sahajbert
pip install -r requirements.txt
pip install -q https://github.com/learning-at-home/hivemind/archive/sahaj2.zip
pip install seqeval
```
###### 2. Run the following command
```
!python train_ner.py \
  --model_name_or_path Upload/sahajbert2 \
  --output_dir sahajbert/ner \
  --learning_rate 1e-5  \
  --max_seq_length 128  \
  --num_train_epochs 20 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 32 \
  --early_stopping_patience 3 \
  --early_stopping_threshold 0.01
```
###### This will give you a prompt, and you need to provide your Huggingface username and password. (We don't store huggingface password) this is only to allow your score to be reflected in the leaderboard.

**Leaderboard link - [sahajBERT2-xlarge-ner](https://wandb.ai/tanmoyio/sahajBERT2-xlarge-ner?workspace=user-tanmoyio)**

If you are using GPU, or finetuning it with colab GPU then you might want to adjust the `per_device_train_batch_size`, `per_device_train_batch_size`.

### NCC

[Will be added soon]
