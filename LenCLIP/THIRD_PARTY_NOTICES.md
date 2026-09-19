# LenCLIP third-party notices

LenCLIP is reconstructed from the source supplied by the user and the supplied manuscript. Renaming the project does not transfer authorship of inherited components. No blanket license is imposed here on third-party code.

## Supplied segmentation baseline

The inherited feature fusion, Transformer segmentation decoder, local pixel refinement, dataset split lists and class-label metadata originate from the supplied implementation associated with:

Bingfeng Zhang, Siyue Yu, Yunchao Wei, Yao Zhao, and Jimin Xiao. **Frozen CLIP: A Strong Backbone for Weakly Supervised Semantic Segmentation.** CVPR 2024, pp. 3796-3806.

[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Frozen_CLIP_A_Strong_Backbone_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2024_paper.pdf)

Retained or modified files include `LenCLIP_model/segformer_head.py`, `LenCLIP_model/Decoder/TransDecoder.py`, `LenCLIP_model/PAR.py`, and the CLIP internals. The supplied source directory contained no top-level license file; this reconstruction does not infer one. Existing copyright and license comments are retained, including the NVIDIA notice in `segformer_head.py`.

## CLIP

The `clip` tokenizer, vocabulary, checkpoint-loading and encoder code derive from OpenAI CLIP through the supplied project, with changes for prompt gradients and dense features.

[OpenAI CLIP repository](https://github.com/openai/CLIP)

The new `LenCLIP_model` prompt, mask and alignment modules use these pretrained components; they do not claim ownership of CLIP or its weights.

## Related components credited by the supplied project

- Lixiang Ru et al. **Learning Affinity from Attention: End-to-End Weakly-Supervised Semantic Segmentation with Transformers.** CVPR 2022. [Project](https://github.com/rulixiang/afa)
- Yuqi Lin et al. **CLIP Is Also an Efficient Segmenter: A Text-Driven Approach for Weakly Supervised Semantic Segmentation.** CVPR 2023. [Project](https://github.com/linyq2117/CLIP-ES)
- SegFormer feature-fusion code retains its original NVIDIA copyright/license notice.

Dataset images and annotations are not redistributed in this archive. The existing split lists and image-level label metadata are retained from the user-supplied project.
