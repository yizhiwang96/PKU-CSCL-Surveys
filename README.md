# PKU-CSCL-Surveys

## Font Generation

 Method  | Publication  | Code | Project | Core Idea | Authors
 ---- | ----- | ------  | ------ | ------ | ------ 
 DeepVecFont  | SIGGRAPH Asia 2021 | [Link](https://github.com/yizhiwang96/deepvecfont) | [Link](https://yizhiwang96.github.io/deepvecfont_homepage/) | Dual-modality learning and Diff Rendering | [Yizhi Wang](https://yizhiwang96.github.io/)
 LF-Font | AAAI 2021 | [Link](https://github.com/clovaai/lffont) | - | components decomposition | [Song Park](https://8uos.github.io/)
 MX-Font | ECCV 2021 | [Link](https://github.com/clovaai/mxfont) | - | multiple experts to represent different local concepts | [Song Park](https://8uos.github.io/)
 MLFont | ICMR 2021 | [Link](https://github.com/Listening33/MLFont) | - | deep meta learning | Xu Chen
 DG-font | CVPR 2021 | [Link](https://github.com/ecnuycxie/DG-Font) | - | Deformable generative networks | Yangchen Xie
 ZiGAN | MM 2021 | - | - | Few-shot calligraphy font generation |Qi Wen, Shuang Li
 
## Handwritting, Sketch and Vectorization

 Method  | Publication  | Code | Project | Core Idea | Authors
 ---- | ----- | ------  | ------ | ------ | ------ 
 SketchRNN  | ICLR 2018 | [Deprecated](https://github.com/hardmaru/sketch-rnn/) [Link](https://github.com/magenta/magenta/blob/main/magenta/models/sketch_rnn/README.md) | [Link](https://experiments.withgoogle.com/sketch-rnn-demo) | Seq2Seq and GMM | [David Ha](https://twitter.com/hardmaru/)
SketchFormer | CVPR 2020 | [Link](https://github.com/leosampaio/sketchformer) | - | Transformers AutoEncoder | [Leo S.F. Ribeiro](https://twitter.com/a_leosampaio)
GVSF | SIGGRAPH 2021 | [Link](https://github.com/MarkMoHR/virtual_sketching) | [Link](https://markmohr.github.io/virtual_sketching/) | CRNN for cropped Windows and then pasting | [Haoran Mo](http://mo-haoran.com/)
Oneshot Sketch Segmentation| [Arxiv](https://arxiv.org/pdf/2112.10838.pdf) | - | - | GCN as encoder, deformation for one shot | [Yulia Gryaditskaya](https://yulia.gryaditskaya.com/)
YOLaT| [NIPS 2021](https://openreview.net/forum?id=_ZXlOpdufFJ)| - | - | Recognizing Vector Graphics without Rasterization | [Xinyang Jiang](https://scholar.google.com/citations?user=JiTfWVMAAAAJ)


## Image and Texture Synthesis
 Method  | Publication  | Code | Project | Core Idea | Authors
 ---- | ----- | ------  | ------ | ------ | ------ 
pix2pixHD  | CVPR 2018 | [Link](https://github.com/NVIDIA/pix2pixHD) | - |  High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs | [Ting-Chun Wang](https://tcwang0509.github.io/) |
SPADE  | CVPR 2019 | [Link](https://github.com/NVlabs/SPADE) | - |  Semantic Image Synthesis with Spatially-Adaptive Normalization | [Taesung Park](https://taesung.me/) |
ADGAN  | CVPR 2020 | [Link](https://github.com/menyifang/ADGAN) | - |  Attribute-Decomposed | [Yifang Men](https://menyifang.github.io/) |
Taming Transformers  | CVPR 2021 | [Link](https://github.com/CompVis/taming-transformers) | - |  VQ-GAN and Transformers | [Björn Ommer](https://ommer-lab.com/people/ommer/) |
CoCosNet-v2  | CVPR 2021 | [Link](https://github.com/microsoft/CoCosNet-v2) | - | full-resolution cross-domain with feature-level PatchMatch  | [Xingran Zhou](https://xingranzh.github.io/), [Bo Zhang](https://bo-zhang.me/) |
ASAPNet | CVPR 2021 | - | [Link](https://tamarott.github.io/ASAPNet_web/) |  Spatially-Adaptive Pixelwise Networks for Fast Image Translation | [Tamar Rott Shaham](https://github.com/tamarott) |
CIPS | CVPR 2021 | [Link](https://github.com/saic-mdal/CIPS) | - |  Image Generators with Conditionally-Independent Pixel Synthesis | [I. Anokhin] |
MaskGIT | - | [Link](https://github.com/saic-mdal/CIPS) | - |  Masked Generative Image Transformer | [Huiwen Chang] |
StyleGAN3 | NeurIPS 2021 | [Link](https://github.com/NVlabs/stylegan3) | - |  Alias-Free Generative Adversarial Networks | [Tero Karras] |
Projected GANs | NeurIPS 2021 | [Link](https://github.com/autonomousvision/projected_gan) | [Link](https://sites.google.com/view/projected-gan) |  Train GANs in pretrained feature spaces | [Axel Sauer](https://axelsauer.com/) |

## Deep Generative models
 Method  | Publication  | Code | Project | Core Idea | Authors
 ---- | ----- | ------  | ------ | ------ | ------ 
DDPM  | - | [Link](https://github.com/hojonathanho/diffusion) | - |  Denoising Diffusion Probabilistic Models | [Jonathan Ho](https://github.com/hojonathanho) |
Flow++  | - | [Link](https://github.com/aravindsrinivas/flowpp) | - |  Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design | [Jonathan Ho](https://github.com/hojonathanho) |
LDMs | arxiv | [Link](https://github.com/CompVis/latent-diffusion) | - | Autoregressive and diffusion models | [Robin Rombach](https://scholar.google.com/citations?user=ygdQhrIAAAAJ&hl=zh-CN&oi=sra)
DenseFlow | NIPS 2021 | - | - | Densely connected normalizing flows | [Matej Grcic]
LSGM | NeurIPS 2021 | - | -  | Score-based Generative Modeling in Latent Space | [Arash Vahdat]
Progressive Distillation | ICLR 2022 | - | -  | PROGRESSIVE DISTILLATION FOR FAST SAMPLING OF DIFFUSION MODELS | [Jonathan Ho](https://github.com/hojonathanho) 

## Attractive Image Generation
 Method  | Publication  | Code | Project | Core Idea | Authors
 ---- | ----- | ------  | ------ | ------ | ------ 
GBCI  | [IEEE TAFFC](https://ieeexplore.ieee.org/document/9353984) | - | - |  Brain-computer interface for generating personally attractive images | [Michiel Spape](http://cognitology.eu/) |
ICE  | [arXiv](https://arxiv.org/abs/2201.09689) | [Link](https://github.com/prclibo/ice) | - |  Interpretable Control Exploration and Counterfactual Explanation on StyleGAN | [Bo Li](http://prclibo.github.io/) |


## Semantic Segmentation
 Method  | Publication  | Code | Project | Core Idea | Authors
 ---- | ----- | ------  | ------ | ------ | ------ 
MaskFormer  | NeurIPS 2021 | [Link](https://github.com/facebookresearch/MaskFormer) | - |  mask classification with Transformers | [Bowen Cheng](https://github.com/bowenc0221) |
Labels4Free  | ICCV 2021 | - | [Link](https://rameenabdal.github.io/Labels4Free/) |  Unsupervised Segmentation using StyleGAN | [Rameen Abdal](https://scholar.google.com/citations?user=kEQimk0AAAAJ&hl=en) |
ProDA  | CVPR 2021 | [Link](https://github.com/microsoft/ProDA) | - |  prototypical pseudo label denoising | [Pan Zhang](https://panzhang0212.github.io/) |
CPS  | CVPR 2021 | [Link](https://github.com/charlesCXK/TorchSemiSeg) | - |  cross pseudo supervision | [Xiaokang Chen](https://charlescxk.github.io/) |
SeMask  | arxiv | [Link](https://github.com/Picsart-AI-Research/SeMask-Segmentation) | - |  add semantic context to pretrained backbones | [Jitesh Jain](https://praeclarumjj3.github.io/) |
MaX-DeepLab  | CVPR 2021 | [Link](https://github.com/google-research/deeplab2) | - |  predict class-labeled masks | Huiyu Wang |
SemiSeg-Contrastive  | ICCV 2021 | [Link](https://github.com/Shathe/SemiSeg-Contrastive) | - |  Pixel-Level Contrastive Learning | Inigo Alonso |
Point Transformer  | ICCV 2021 | - | - |  transformer for point cloud processing | Hengshuang Zhao |
Self-training  | NeurIPS 2020 | [Link](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/self_training) | - |   self-training | Barret Zoph |

## 3D Vision
### Novel View Synthesis (Image Inpainting)
 Method  | Publication  | Code | Project | Core Idea | Authors
 ---- | ----- | ------  | ------ | ------ | ------ 
 3D Ken  | SIGGRAPH 2019 | [Link](https://github.com/sniklaus/3d-ken-burns) | - | RGBD Inpainting and Point Cloud Rendering | [Simon Niklaus](http://sniklaus.com/welcome)
 3D Photo  | CVPR 2020 | [Link](https://github.com/vt-vl-lab/3d-photo-inpainting) | [Link](https://shihmengli.github.io/3D-Photo-Inpainting/)  | Layered RGBD Inpainting | [Meng-Li Shih](https://shihmengli.github.io/) 
 ### Novel View Synthesis (NeRF-alike)
 Method  | Publication  | Code | Project | Core Idea | Authors
 ---- | ----- | ------  | ------ | ------ | ------ 
 Neural Volumes | SIGGRAPH 2019 | [Link](https://github.com/facebookresearch/neuralvolumes) | [Link](https://stephenlombardi.github.io/projects/neuralvolumes/) | Neural Volume Rendering | [Stephen Lombardi](https://stephenlombardi.github.io/)
 NeRF | ECCV 2020 | [Link](https://github.com/bmild/nerf) | [Link](https://www.matthewtancik.com/nerf) | Radiance Field for Volume Rendering | [Ben Mildenhall](https://bmild.github.io/) 
 NeX | CVPR 2021 | [Link](https://github.com/nex-mpi/nex-code/) | [Link](https://nex-mpi.github.io/) | - | Suttisak Wizadwongsa
 Nerfies | ICCV 2021 | [Link](https://github.com/google/nerfies) | [Link](https://nerfies.github.io/) | - | [Keunhong Park](https://keunhong.com/)
 KiloNeRF | ICCV 2021 | [Link](https://github.com/creiser/kilonerf) | [Link](https://creiser.github.io/kilonerf/) | thousands of tiny MLPs | [Songyou Peng](https://pengsongyou.github.io/)
  Neural Actor | SIGGRAPH Asia 2021 | - | [Link](https://vcai.mpi-inf.mpg.de/projects/NeuralActor/) | pose(SMPL)-guided view synthesis | [Lingjie Liu](https://lingjie0206.github.io/)
 Light Field NR | Arxiv | [Link](https://github.com/google-research/google-research/tree/master/light_field_neural_rendering) | [Link](https://light-field-neural-rendering.github.io/?s=05) | - | [Mohammed Suhail](https://mohammedsuhail.net/)
 Plenoxels | Arxiv | [Link](https://github.com/sxyu/svox2) | [Link](https://alexyu.net/plenoxels/) | 3D grid with spherical harmonics,without neural network | [Alex Yu](https://alexyu.net/)
 ING | Arxiv 2022 | [Link](https://nvlabs.github.io/instant-ngp/) | [Link](https://github.com/NVlabs/instant-ngp) | - | [Thomas Müller](https://tom94.net/)
  ### 3D-aware Image Synthesis (NeRF + GAN)
   Method  | Publication  | Code | Project | Core Idea | Authors
 ---- | ----- | ------  | ------ | ------ | ------ 
  pi-GAN | CVPR 2021(Oral) | [Link](https://github.com/marcoamonteiro/pi-GAN) | [Link](https://marcoamonteiro.github.io/pi-GAN-website/) | represent scenes as view-consistent radiance fields | [Eric Chan](https://ericryanchan.github.io/)
  GIRAFFE | CVPR 2021(Oral) | [Link](https://github.com/autonomousvision/giraffe) | [Link](https://m-niemeyer.github.io/project-pages/giraffe/index.html) | representing scenes as compositional generative neural feature fields (same as pi-GAN?) but can disentangle individual objects and do some edits | [Michael Niemeyer](https://m-niemeyer.github.io)
 GRAM  | Arxiv | - | [Link](https://yudeng.github.io/GRAM/) | Isosurfaces For Point Sampling | [Yu Deng](https://yudeng.github.io/)
  EG3D | Arxiv | [Link](https://github.com/NVlabs/eg3d) | [Link](https://matthew-a-chan.github.io/EG3D/?s=05) | hybrid explicit–implicit tri-plane representation  | [Eric Ryan Chan](https://ericryanchan.github.io/)
CIPS-3D  | Arxiv | [Link](https://github.com/PeterouZh/CIPS-3D) | - | style and NeRF-based generator | Peng Zhou




## Implict function

 Method  | Publication  | Code | Project | Core Idea | Authors
 ---- | ----- | ------  | ------ | ------ | ------ 
 DeepSDF | CVPR 2019 | [Link](https://github.com/facebookresearch/DeepSDF) | [Link](https://github.com/facebookresearch/DeepSDF) | AutoDecoder SDF | [Jeong Joon Park](https://scholar.google.com/citations?user=2feSMg8AAAAJ&hl=zh-CN&oi=sra)
 IM-Net | CVPR 2019 | [Link](https://github.com/czq142857/implicit-decoder) | [Link](https://www.sfu.ca/~zhiqinc/imgan/Readme.html) | Implicit Field Decoder | [Zhiqin Chen](https://www.sfu.ca/~zhiqinc/)
 BSPNet  | CVPR 2020 | [Link](https://github.com/czq142857/BSP-NET-original) | [Link](https://github.com/czq142857/BSP-NET-original) | Binary Space Partitioning | [Zhiqin Chen](https://www.sfu.ca/~zhiqinc/)
 
 
## Diffentiable Rendering

 Method  | Publication  | Code | Project | Core Idea | Authors
 ---- | ----- | ------  | ------ | ------ | ------ 
 DiffVG  | Siggraph Asia 2020 | [Link](https://github.com/BachiLi/diffvg) | [Link](https://people.csail.mit.edu/tzumao/diffvg/) | Monte Carlo sampling | [Tzu-Mao Li](https://people.csail.mit.edu/tzumao/)
 
## Scene Text Detection and Recognition
 Method  | Publication  | Code | Project | Core Idea | Authors
 ---- | ----- | ------  | ------ | ------ | ------ 
 RPI | WACV 2022 | - | - | Rectifying Principle Irregularities | [Fan Bai](https://scholar.google.com/citations?user=o-6RFGEAAAAJ&hl=en&oi=ao)
 
 ## Multimodal Synthesis
  Method  | Publication  | Code | Project | Core Idea | Authors
 ---- | ----- | ------  | ------ | ------ | ------ 
  GLIDE  | [arXiv](https://arxiv.org/pdf/2112.10741.pdf) | [Link](https://github.com/openai/glide-text2im) | [demo](https://huggingface.co/spaces/valhalla/glide-text2im) | Text-guided image generation and editing with diffusion models | [Alex Nichol](https://aqnichol.com/)
