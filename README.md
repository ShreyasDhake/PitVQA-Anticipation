# Anticipating Surgical Events via GRU-Gated Temporal Cross-Attention in Video Question Answering

## Abstract: 
Anticipating future events during surgery is essential for real-time, intelligent assistance in high-risk procedures such as endonasal transsphenoidal pituitary surgery, where surgeons must navigate complex anatomy under limited visibility. While recent Visual Question Answering (VQA) models show promise in surgical applications, most rely on isolated frames and static vision-language alignment, lacking the temporal reasoning needed to forecast upcoming steps or instrument usage. Furthermore, existing surgical VQA datasets primarily focus on questions grounded in the current surgical context, offering limited support for anticipatory understanding. To address these gaps, we introduce PitVQA-Anticipation, the first video-based VQA dataset explicitly designed for forward-looking surgical reasoning. Spanning 33.5 hours of high-resolution pituitary surgery videos and comprising 734,769 question–answer pairs, the dataset covers four key anticipatory tasks: predicting future phases, steps, instruments, and remaining duration. Each QA pair is generated from temporally grouped frames and curated with expert surgical annotation, making it a rich resource for time-sensitive multimodal learning. Building on this dataset, we propose SurgicalViVQA, a novel video-language architecture that adapts large language models (LLMs) for surgical video understanding. At its core is a GRU-Gated Temporal Cross-Attention module that introduces two technical innovations: (i) a bidirectional GRU-based video encoder to capture temporal dependencies across frames, and (ii) an adaptive gating mechanism that dynamically controls the injection of visual context into the language stream at the token level. This design enables fine-grained, context-aware reasoning over evolving surgical scenes. Experimental results show that SurgicalViVQA significantly outperforms strong baselines and state-of-the-art models on PitVQA-Anticipation, setting a new benchmark for future-aware surgical AI.

## PitVQA-Anticipation Dataset:
Dataset will be released upon the paper acceptance

## SurgicalViVQA Pretrained Weights
The pretrained weights will be releases upon the acceptance of the paper

## Training Command
```
python main.py
```

## Inference Command
```
python inference.py
```

## Acknowlwdgement
The implementation of SurgicalViVQA relies on resources from  <a href="https://github.com/huggingface/transformers">Huggingface Transformers</a>, <a href="https://github.com/huggingface/peft">PEFT</a>, <a href="https://github.com/xuguohai/X-CLIP">X-CLIP</a>. We thank the original authors for their open-sourcing. 
