# CHIME: Cross-passage Hierarchical Memory Network for Generative Review Question Answering
This repository contains PyTorch implementation of the [corresponding COLING 2020 Paper](wait for adding).

## Breif Introduction
CHIME is a cross-passage hierarchical memory network for generative question answering (QA). It extends [XLNet](https://github.com/zihangdai/xlnet) introducing an auxiliary memory module consisting of two components: the **context memory** collecting cross-passage evidences, and the **answer memory** working as a buffer continually refining the generated answers. 

The following syntactically well-formed answers show the efficacy of CHIME.
- *Question1: can this chair be operated with battery only?*
- *yes, it can be operated by battery, but it is not recommended to use this chair with batteries only*
