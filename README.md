# SENSE-2ATTR: Unified Speech Encoder for Multiple Attributes

**This repository provides the training code for SENSE-2ATTR**, a unified speech encoder that jointly learns multiple **utterance-level attribute
representations** from a single speech signal.

This version learns two attributes:

- **Semantic representation:** 1024 dimensions.
- **Speaker representation:** 192 dimensions.

[Paper](https://arxiv.org/abs/2603.08312) · [Pretrained model on Hugging Face](https://huggingface.co/LIA-AvignonUniversity/SENSE-2ATTR)

## Teacher models

Each attribute is learned through distillation from a pretrained teacher:

| Attribute | Teacher model | Dimension |
| --- | --- | --- |
| Semantic | [BGE-M3](https://huggingface.co/BAAI/bge-m3) | 1024 |
| Speaker | [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | 192 |

BGE-M3 provides semantic targets from the transcripts, while ECAPA-TDNN provides
speaker targets from the audio. Both teachers remain frozen during training.

The shared speech encoder is jointly fine-tuned with two attribute-specific
branches. Each branch learns its own per-layer projections, layer-mixing weights,
and attention pooling.

## Data

SENSE-2ATTR was trained on the multilingual **Common Voice** dataset, combining
the following **90 languages** into a single multilingual training setup:

`af`, `am`, `ar`, `as`, `ast`, `az`, `ba`, `be`, `bg`, `bn`,
`br`, `ca`, `ckb`, `cs`, `cv`, `cy`, `da`, `de`, `dv`, `el`,
`en`, `eo`, `es`, `et`, `fa`, `fi`, `fr`, `fy-NL`, `ga-IE`, `gl`,
`gn`, `he`, `hi`, `hsb`, `ht`, `hu`, `ia`, `id`, `is`, `it`,
`ja`, `ka`, `kab`, `kk`, `ko`, `ky`, `lt`, `lo`, `lv`, `ml`,
`mn`, `mhr`, `mk`, `mr`, `mt`, `ne-NP`, `nl`, `nn-NO`, `oc`, `or`,
`os`, `pa-IN`, `pl`, `ps`, `pt`, `ro`, `ru`, `sah`, `sc`, `sk`,
`sl`, `sr`, `sv-SE`, `sw`, `ta`, `te`, `th`, `ti`, `tk`, `tr`,
`tt`, `ug`, `uk`, `ur`, `uz`, `vi`, `yi`, `yo`, `zh-HK`, `zu`.

### Training manifests

The training recipe reads two prepared manifests, **`train.csv`** and
**`dev.csv`**, referencing mono audio at **16 kHz**.

For data preparation, see SpeechBrain's
[Common Voice SENSE preparation script](https://github.com/speechbrain/speechbrain/blob/develop/recipes/CommonVoice/common_voice_sense_prepare.py).
Prepare the audio and manifests before running `train.py`.

## Installation

Install SpeechBrain and Transformers in an environment with matching PyTorch
and Torchaudio builds:

```bash
pip install speechbrain==1.1.0 transformers
```

For training, install the additional data and teacher-model dependencies:

```bash
pip install pandas hyperpyyaml FlagEmbedding flair spacy "ruamel.yaml==0.18.6"
```

## Training

Run the following command from the repository root:

```bash
python train.py hparams/train_2attr.yaml
```

Model sources, batch sizes, learning rates, and the output directory are
configured in [`hparams/train_2attr.yaml`](hparams/train_2attr.yaml).

## Extract representations with the pretrained model

The pretrained **SENSE-2ATTR** model is available on
[Hugging Face](https://huggingface.co/LIA-AvignonUniversity/SENSE-2ATTR).

Load the model with SpeechBrain:

```python
from speechbrain.inference.interfaces import foreign_class

model = foreign_class(
    source="LIA-AvignonUniversity/SENSE-2ATTR",
    pymodule_file="custom.py",
    classname="SENSEEncoder",
    run_opts={"device": "cuda:0"},
)
```

Extract both representations from an audio file:

```python
outputs = model.encode_file("example.wav")

semantic_embedding = outputs["sem"]
# torch.Size([1, 1024])

speaker_embedding = outputs["spkr"]
# torch.Size([1, 192])
```

## Paper

**Learning Multiple Utterance-Level Attribute Representations with a Unified Speech Encoder**

Maryem Bouziane, Salima Mdhaffar, Yannick Estève

[Read the paper on arXiv](https://arxiv.org/abs/2603.08312).
