---
title: "Books and Reading Guide"
subtitle: "A Structured Study Roadmap for Image Denoising with Residual Networks"
author: "Adebanji Oluwatimileyin Adelowo"
date: "2026"
toc: true
toc-depth: 3
number-sections: true
geometry: margin=1in
fontsize: 11pt
linkcolor: blue
urlcolor: blue
header-includes: |
  \usepackage{fvextra}
  \DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,breakanywhere,breaksymbolleft={},commandchars=\\\{\}}
  \usepackage{booktabs}
  \setlength{\emergencystretch}{4em}
  \sloppy
---

# How to Use This Guide

## What this document is, and what it is not

This is a **study roadmap**, not a bibliography. The companion document,
*Deep Residual Learning for Image Denoising* (referred to throughout as **the
technical document**), derives the mathematics behind the two notebooks in the
`Image_denoising_using_ResNet` repository and cites 54 research papers. Papers
are the right medium for a specific result, and a poor medium for learning a
subject: they assume the background rather than supply it, they compress
derivations, and they are written for people who already know why the problem
matters.

Books supply what papers omit. This guide identifies **14 books** that between
them cover every mathematical object the technical document uses, and for each
one it states:

* the exact bibliographic identity (edition, publisher, year, ISBN, DOI or
  publisher page), verified against the publisher's own record;
* **why it is relevant to this project specifically** --- not to image
  processing in general;
* **which chapters to read**, and which to skip;
* **what to extract** from those chapters --- the specific result, derivation,
  or intuition that pays off;
* **how it connects to the actual code** in the two notebooks;
* its **level** and its **classification** as foundational, core, supplementary,
  or advanced;
* its **relationship to the papers** already cited in the technical document ---
  which citation it provides the background for.

A note on chapter numbering: it can shift between editions and printings. Where
a chapter number is given below it corresponds to the edition named in that
entry; readers using a different edition should confirm against their own copy's
table of contents, which is why each chapter reference is given with its title as
well as its number.

## The gap this project sits in

The repository trains two convolutional residual networks to remove additive
white Gaussian noise from grayscale LFW face images. Understanding that work
properly requires four distinct bodies of knowledge, and no single book covers
more than two of them:

1. **Where noise comes from and what it is.** Sensor physics, the
   Poisson--Gaussian model, and the idealization to AWGN. (Technical document
   Sections 1.2 and 2.1.)
2. **What "estimating a clean image" means.** MAP versus MMSE estimation,
   ill-posedness, regularization, and Tweedie's formula --- which is what makes
   the global residual skip connection the *principled* architecture rather than
   a trick. (Sections 2.2--2.4, 3.3.)
3. **What the classical methods were, and why they stopped improving.** Spatial
   filtering, anisotropic diffusion, total variation, wavelet shrinkage,
   non-local means, BM3D. (Section 4.)
4. **How convolutional networks work and why residual learning and channel
   attention help.** (Sections 3.3--3.4, 5, 6, 7.)

Reading only deep-learning material teaches you to build the enhanced notebook
without understanding why its architecture takes the form it does. Reading only
classical material leaves you unable to read the code. The reading order in
Section 2 is designed to close both gaps in the right sequence.

## Three routes through the material

Not everyone needs all fourteen books. Pick a route.

**Route A --- "I need to understand and extend this code" (about 6 weeks).**
Kok & Tam (Ch. 1--3, 5--6) $\to$ Goodfellow et al. (Ch. 5--9) $\to$ Prince
(Ch. 10--11) $\to$ Wang & Bovik. Four books, minimum theory, maximum direct
relevance. This route gets you to the point where every line of both notebooks
is intelligible and you can run the ablation grid proposed in Section 12.1 of
the technical document.

**Route B --- "I want the theory properly" (one semester).** Route A, preceded
by Kay (Ch. 10--12) and followed by Chan & Shen (Ch. 4), Mallat (Ch. 11), and
Hansen (Ch. 1--5). Seven books. This is the route that makes the technical
document's Sections 2 and 4 feel obvious rather than asserted.

**Route C --- "I am starting research in image restoration" (two semesters).**
All fourteen, in the order given in Section 2, with Aubert & Kornprobst, Elad,
and Barbu read as reference works rather than cover to cover.

# Reading Order at a Glance

| # | Book (short) | Tier | Read for | Technical-doc sections |
|---|---|---|---|---|
| 1 | Gonzalez & Woods, *Digital Image Processing* | Foundational | Vocabulary, noise models, spatial filtering, PSNR | 1.2, 3.1, 3.5, 4.1 |
| 2 | Kay, *Estimation Theory* | Foundational | MAP, MMSE, the Bayesian frame | 2.2, 2.3, 7.1 |
| 3 | Kok & Tam, *Digital Image Denoising in MATLAB* | Core | The whole classical pipeline, implemented | 4 (all) |
| 4 | Goodfellow et al., *Deep Learning* | Core | Convolution, optimization, regularization, BN | 3.1, 3.4, 7.5--7.6 |
| 5 | Prince, *Understanding Deep Learning* | Core | Residual networks, attention, diffusion | 3.3, 5.4--5.6, 6 |
| 6 | Wang & Bovik, *Modern Image Quality Assessment* | Core | SSIM, from the people who invented it | 3.6, 7.3--7.4 |
| 7 | Chan & Shen, *Image Processing and Analysis* | Core | TV/ROF, PDE and wavelet denoising unified | 4.2--4.4 |
| 8 | Mallat, *A Wavelet Tour* | Core | Wavelet shrinkage, minimax optimality | 4.4 |
| 9 | Hansen, *Discrete Inverse Problems* | Supplementary | Ill-posedness, regularization, parameter choice | 2.4 |
| 10 | Boyd & Vandenberghe, *Convex Optimization* | Supplementary | Proximal operators, projection, convexity | 3.5, 4.4, 7 |
| 11 | Szeliski, *Computer Vision* | Supplementary | Where denoising sits in a vision pipeline | 1.2, 4, 5 |
| 12 | Elad, *Sparse and Redundant Representations* | Advanced | Sparsity, K-SVD, the BM3D worldview | 4.5--4.6 |
| 13 | Aubert & Kornprobst, *Mathematical Problems in Image Processing* | Advanced | Rigorous PDE/variational analysis | 4.2--4.3 |
| 14 | Barbu, *Novel Diffusion-Based Models* | Advanced | Modern nonlinear diffusion variants | 4.2 |

The tiers are cumulative: nothing in Tier "Core" assumes anything not in Tier
"Foundational", and the Advanced entries assume the Core.

# Tier 1 --- Foundational

These two books establish the vocabulary and the estimation-theoretic frame.
Read them first; everything afterwards assumes them.

## Gonzalez & Woods, *Digital Image Processing*

| Field | Value |
|---|---|
| Authors | Rafael C. Gonzalez, Richard E. Woods |
| Edition | 4th edition |
| Publisher | Pearson |
| Year | 2018 |
| ISBN | 978-0-13-335672-4 |
| Pages | 1192 |
| Level | Undergraduate / early graduate |
| Classification | **Foundational** |

**Why it is relevant to this project.** This is the book that defines the terms
the technical document uses without ceremony: image as a sampled and quantized
2-D function, intensity normalization to $[0,1]$, spatial filtering as
correlation with a kernel, padding, and the taxonomy of noise. Before you can
appreciate that the notebooks' `add_gaussian_noise` implements one specific and
rather idealized corruption model, you need to have seen the others --- Rayleigh,
gamma, exponential, uniform, and impulse (salt-and-pepper) --- laid out with
their densities side by side. Gonzalez & Woods is where that comparison lives,
and it is also where the median filter is developed, which matters because the
notebooks apply a Gaussian-trained network to salt-and-pepper noise and report
$22.56$ dB without comparing against the matched estimator (technical document
Section 10.3).

**Topics it covers that appear in the technical document.**

* Noise probability density functions and how to estimate noise parameters from
  a flat image patch (technical document Sections 1.2 and 8.2, where the network
  must infer $\sigma$ internally).
* Spatial filtering: linear smoothing, the bias--variance trade-off of averaging,
  and order-statistic filters (Section 4.1).
* Frequency-domain view of smoothing and why noise occupies the high band
  (Section 3.3.2, the argument for residual learning).
* Restoration in the presence of noise only, and Wiener filtering (the classical
  ancestor of BM3D's second stage, Section 4.6).
* PSNR and MSE as fidelity measures (Section 3.5).

**Chapters to study.**

* **Chapter 2, *Digital Image Fundamentals*** --- sampling, quantization,
  intensity, and the coordinate conventions. Skim if already comfortable.
* **Chapter 3, *Intensity Transformations and Spatial Filtering*** --- read
  carefully. This is the mathematical content behind `Conv2D`, including
  correlation versus convolution (the technical document's convolution equation is a correlation, as Gonzalez & Woods explains), kernel
  separability, and padding.
* **Chapter 4, *Filtering in the Frequency Domain*** --- read the sections on
  lowpass and highpass filtering. This gives the spectral intuition for why an
  image's energy is concentrated at low frequencies while AWGN is flat, which is
  the core of the residual-learning argument.
* **Chapter 5, *Image Restoration and Reconstruction*** --- **the most important
  chapter in this book for this project.** Read the noise-model section, the
  spatial-filtering-for-noise-only section (mean filters, order-statistic
  filters, adaptive filters), and the Wiener filtering section.
* **Chapter 7, *Wavelet and Other Image Transforms*** --- read as a gentle
  preparation for Mallat.
* Skip: compression, morphology, segmentation, and representation chapters, none
  of which touch this project.

**What to extract.** Three things. (i) The full table of noise models, so that
"AWGN" registers as *one choice among many*. (ii) The derivation of why an
$m \times m$ mean filter divides noise standard deviation by $m$ while blurring
edges --- this is the tension every method in the technical document's Section 4
tries to resolve. (iii) The adaptive local-noise-reduction filter, which
modulates its strength by the ratio of local variance to noise variance; it is
the classical antecedent of what a blind CNN denoiser must learn to do
implicitly.

**Connection to the code.** `images_clean = color.rgb2gray(...)` and the
normalization to $[0,1]$ are Chapter 2 material. The $3\times3$ convolutions with
`padding='same'` are Chapter 3. `add_salt_and_pepper_noise` and the evaluation in
cell 18/19 are Chapter 5.

**Relationship to the cited papers.** Provides the background for essentially
every classical citation in the technical document's Section 4, and specifically
prepares the reader for Perona & Malik (1990) and Rudin, Osher & Fatemi (1992) by
first establishing what a linear filter can and cannot do.

## Kay, *Fundamentals of Statistical Signal Processing, Volume I: Estimation Theory*

| Field | Value |
|---|---|
| Author | Steven M. Kay |
| Volume | Volume I: Estimation Theory |
| Publisher | Prentice Hall |
| Year | 1993 |
| ISBN | 978-0-13-345711-7 |
| Pages | 595 |
| Level | Graduate |
| Classification | **Foundational** |

**Why it is relevant to this project.** The technical document's Section 2 claims
that denoising *is* an estimation problem, that MAP and MMSE are two different
answers to it, and that the MMSE estimator is the posterior mean --- which is
then used in Section 7.1 to explain why MSE-trained denoisers blur. Those are not
image-processing facts; they are estimation-theory facts, and Kay is the standard
reference that proves them. If you read one thing to make the technical document
rigorous rather than plausible, read Kay's Bayesian chapters.

**Topics it covers that appear in the technical document.**

* The classical/Bayesian distinction, and why a prior is *required* when the
  likelihood alone has no useful maximum (technical document Section 2.4).
* MMSE estimation and the theorem that the minimum mean-square-error estimator is
  the conditional mean $\mathbb{E}[x \mid y]$ (the MMSE equation). This is
  the single most load-bearing result borrowed from outside image processing in
  the whole technical document.
* MAP estimation and its relationship to MMSE (they coincide for Gaussian
  posteriors, and diverge exactly where the posterior is skewed or multimodal ---
  which is the blur argument of Section 7.1).
* Linear Bayesian estimation and the Wiener filter, which is what BM3D's second
  stage computes (Section 4.6).
* The bias--variance decomposition and the notion of an estimator's risk.

**Chapters to study.**

* **Chapter 10, *The Bayesian Philosophy*** --- read in full. It sets up prior,
  posterior, and the difference between treating $x$ as deterministic-unknown
  versus random.
* **Chapter 11, *General Bayesian Estimators*** --- **the essential chapter.**
  It derives the MMSE estimator as the posterior mean, the MAP estimator as the
  posterior mode, and compares them. Everything in the technical document's
  Sections 2.2, 2.3 and 7.1 is a specialization of this chapter to
  $y = x + n$.
* **Chapter 12, *Linear Bayesian Estimators*** --- the LMMSE/Wiener material.
  Read at least the scalar and vector Gaussian cases.
* **Chapters 2--4** (minimum variance unbiased estimation, the Cramér--Rao lower
  bound, linear models) --- read if you want the classical half of the picture,
  particularly the CRLB as a statement about fundamental limits, which is the
  estimation-theoretic cousin of Levin & Nadler's (2011) denoising bounds.
* **Chapter 7, *Maximum Likelihood Estimation*** --- useful context; explains why
  $\hat{x} = y$ is the ML answer for the denoising likelihood and hence why ML
  alone is useless here.
* Skip: the signal-processing application chapters (spectral estimation, array
  processing) unless they interest you independently.

**What to extract.** The proof that $\arg\min_f \mathbb{E}\|x - f(y)\|^2 = \mathbb{E}[x\mid y]$,
and the *intuition* for what a posterior mean does when the posterior is
multimodal. Once you have internalized that a squared-error-optimal estimator
must average over all plausible reconstructions, the blurriness of MSE-trained
networks stops being a mysterious empirical fact and becomes a theorem.

**Connection to the code.** The baseline notebook's
`loss='mean_squared_error'` is a decision to compute $\mathbb{E}[x \mid y]$. The
enhanced notebook's `combined_loss` is a decision to compute something else on
purpose. Kay is where you learn what that difference means.

**Relationship to the cited papers.** Supplies the background for Robbins (1956),
Miyasawa (1961) and Efron (2011) --- the Tweedie's-formula lineage in Section 2.3
--- and for the Blau & Michaeli (2018) perception--distortion trade-off, which is
a statement about the geometry of estimators that only makes sense once Kay's
Chapter 11 is understood.

# Tier 2 --- Core

These five books are the heart of the guide. Together they cover the classical
denoising pipeline, the deep-learning machinery, and the quality metrics.

## Kok & Tam, *Digital Image Denoising in MATLAB*

| Field | Value |
|---|---|
| Authors | Chi-Wah Kok, Wing-Shan Tam |
| Publisher | Wiley--IEEE Press |
| Year | 2024 |
| ISBN | 978-1-119-61769-3 |
| DOI | 10.1002/9781119617778 |
| Level | Advanced undergraduate / graduate, implementation-oriented |
| Classification | **Core** --- the single most on-topic book in this guide |

**Why it is relevant to this project.** This is the only book in the list whose
subject is *exactly* the subject of the repository. Its seven chapters march
through the same sequence as the technical document's Section 4 --- filtering,
wavelets, variational methods, non-local means --- and it supplies runnable
MATLAB source for every algorithm, which means you can generate the classical
baselines that the notebooks conspicuously lack. The technical document's
Section 11.8 lists "no classical or literature baselines" as a limitation; this
book is the most direct route to fixing it.

**Verified table of contents.**

| Ch. | Title | Pages |
|---|---|---|
| 1 | Digital Image | 1--46 |
| 2 | Filtering | 47--77 |
| 3 | Wavelet | 79--115 |
| 4 | Rank Minimization | 117--136 |
| 5 | Variational Method | 137--147 |
| 6 | NonLocal Means | 149--168 |
| 7 | Random Sampling | 169--182 |

**Chapters to study, and what each maps onto.**

* **Chapter 1, *Digital Image*** --- image representation, noise models, and
  quality metrics. Read alongside the technical document's Sections 2.1 and 3.5.
* **Chapter 2, *Filtering*** --- linear and non-linear spatial filtering. Maps
  onto technical document Section 4.1. Extract: the quantitative
  noise-reduction-versus-blur trade-off, and the median filter, which is the
  correct comparison for the salt-and-pepper experiment in the notebooks'
  cell 18/19.
* **Chapter 3, *Wavelet*** --- transform-domain thresholding. Maps onto
  Section 4.4. Extract: a working soft-thresholding implementation, and the
  practical business of threshold selection, which the technical document treats
  only theoretically via the universal threshold $\sigma\sqrt{2\log N}$.
* **Chapter 4, *Rank Minimization*** --- low-rank models (WNNM and relatives).
  This is the one classical family the technical document does **not** cover, and
  it is worth reading precisely for that reason: grouping similar patches into a
  matrix and shrinking its singular values is a close cousin of BM3D's
  collaborative filtering, and it was competitive with early CNN denoisers.
* **Chapter 5, *Variational Method*** --- the ROF/TV material. Maps onto
  Section 4.3. Extract: a numerical solver for the ROF problem, since the
  technical document gives the Euler--Lagrange equation but not an algorithm.
* **Chapter 6, *NonLocal Means*** --- maps onto Section 4.5. Extract: the
  practical parameterization (search-window size, patch size, filtering
  parameter $h$) and the $2\sigma^2$ distance correction discussed in the
  technical document.
* **Chapter 7, *Random Sampling*** --- acceleration strategies; read if the
  $O(HW S^2 P^2)$ cost of NLM is a practical concern for you.

**What to extract overall.** A set of working classical baselines and a feel for
their parameter sensitivity. The most valuable exercise this book enables: run
NLM, TV, and wavelet shrinkage on the same 1,000 LFW test images at
$\sigma = 0.09$ and put their PSNR/SSIM into the technical document's results
table alongside the $21.42 / 28.32 / 33.71$ dB rows. Only then do the network
numbers acquire absolute meaning.

**Connection to the code.** Chapter 1's metric definitions correspond directly to
the notebooks' `PSNR` and `SSIM` helper functions; Chapter 6 corresponds to the
self-similarity intuition that a CNN's receptive field only partially replicates.

**Relationship to the cited papers.** It is the implementation companion to
Buades et al. (2005), Rudin, Osher & Fatemi (1992), Donoho & Johnstone (1994),
and Dabov et al. (2007) --- four of the technical document's core classical
citations.

## Goodfellow, Bengio & Courville, *Deep Learning*

| Field | Value |
|---|---|
| Authors | Ian Goodfellow, Yoshua Bengio, Aaron Courville |
| Publisher | MIT Press |
| Year | 2016 |
| ISBN | 978-0-262-03561-3 |
| Free online | `https://www.deeplearningbook.org` |
| Level | Graduate |
| Classification | **Core** |

**Why it is relevant to this project.** Every non-architectural decision in the
enhanced notebook --- Adam, He initialization, batch normalization, the
regularizing effect of data augmentation, early stopping, the reason mini-batch
size interacts with gradient noise --- is treated here with more care than any
paper gives it. The technical document states these things; Goodfellow et al.
justify them.

**Topics it covers that appear in the technical document.**

* Convolution as an operation: sparse interactions, parameter sharing, and
  equivariance to translation. This is precisely the argument the technical
  document's Section 3.1 uses to license patch training.
* Optimization for deep models: why SGD-family methods work, learning-rate
  schedules, and adaptive methods including Adam (Section 7.6).
* Regularization: weight decay, early stopping *as* regularization, dataset
  augmentation, and noise robustness --- the last of these is directly the
  enhanced notebook's resampled-noise strategy (Section 8.2).
* Batch normalization, presented as a reparameterization (Section 3.4).
* Maximum likelihood, MSE, and cross-entropy as instances of the same principle,
  which frames the loss-function discussion of Section 7.

**Chapters to study.**

* **Chapter 5, *Machine Learning Basics*** --- read the sections on capacity,
  overfitting, the bias--variance trade-off, maximum likelihood, and Bayesian
  statistics. This is the bridge from Kay to neural networks.
* **Chapter 6, *Deep Feedforward Networks*** --- architecture design,
  hidden units (including why ReLU), and back-propagation.
* **Chapter 7, *Regularization for Deep Learning*** --- read in full. Sections on
  dataset augmentation, noise robustness, and early stopping are all directly
  instantiated in the notebooks.
* **Chapter 8, *Optimization for Training Deep Models*** --- read in full. Covers
  parameter initialization strategies (including the variance-preserving argument
  behind `he_normal`), adaptive learning rates and Adam, and batch normalization.
* **Chapter 9, *Convolutional Networks*** --- read in full. Convolution, pooling,
  variants, and the structured-output setting that denoising belongs to.
* Skip on a first pass: Part III (deep generative models) except as noted under
  Prince below; the practical-methodology chapter is worth a skim.

**What to extract.** The parameter-initialization argument (why variance
$2/\mathrm{fan\_in}$ for ReLU), the treatment of *noise injection as
regularization*, and a clear understanding of why mini-batch gradient estimates
have variance that depends on batch composition --- which is the technical
document's Section 8.1 argument that a batch of random crops is a better gradient
estimator than a batch of near-identical portraits.

**Connection to the code.** `kernel_initializer='he_normal'`,
`keras.optimizers.Adam`, `BatchNormalization()`, `EarlyStopping`, and the
augmentation inside `augment_and_noise` are all chapters 7--9 material.

**Relationship to the cited papers.** Provides the textbook treatment behind
Ioffe & Szegedy (2015), Kingma & Ba (2015), and He et al. (2015). It predates and
therefore does not cover ResNets in depth --- for that, use Prince.

## Prince, *Understanding Deep Learning*

| Field | Value |
|---|---|
| Author | Simon J. D. Prince |
| Publisher | MIT Press |
| Year | 2023 |
| ISBN | 978-0-262-04864-4 |
| Pages | 544 |
| Free draft | `https://udlbook.github.io/udlbook/` |
| Level | Graduate, but unusually readable |
| Classification | **Core** |

**Why it is relevant to this project.** This is the modern complement to
Goodfellow et al., and it covers the three things that book cannot: **residual
networks**, **attention**, and **diffusion models**. Those are, respectively, the
architecture of both notebooks, the mechanism of the Squeeze-and-Excitation
block, and the reason the technical document's Tweedie's-formula discussion
matters beyond denoising. It is the single best textbook treatment of the
material in the technical document's Sections 3.3, 5.4--5.6 and 6.

**Topics it covers that appear in the technical document.**

* Residual connections: why they ease optimization, the shattered-gradients
  view, and their interaction with batch normalization. Directly Section 3.3.
* Batch normalization revisited with the modern (loss-landscape) explanation
  rather than the internal-covariate-shift story --- matching the technical
  document's Section 3.4 caveat.
* Attention and transformers, which is the general form of which
  Squeeze-and-Excitation is the cheapest special case (Section 6.3).
* Diffusion models, whose training objective is a denoising regression and whose
  connection to denoising runs through exactly the identity in Section 2.3.

**Chapters to study.**

* **Chapter 10, *Convolutional Networks*** --- read for the modern presentation
  and for receptive-field reasoning, which the technical document uses
  quantitatively in Section 3.2.
* **Chapter 11, *Residual Networks*** --- **the essential chapter for this
  project.** It develops residual connections, explains the gradient-flow
  argument that the technical document gives as the gradient-flow equation, and
  covers batch normalization in the residual context.
* **Chapter 12, *Transformers*** --- read the attention sections. You are looking
  for the general pattern: compute a data-dependent weighting, apply it
  multiplicatively. Once that is clear, SE reads as "attention with a globally
  pooled query and a diagonal output", and Restormer's channel attention reads as
  the full-rank version.
* **Chapter 18, *Diffusion Models*** --- read after the technical document's
  Section 2.3. This is where "a denoiser is a score estimator" becomes a
  constructive algorithm rather than an observation.
* Chapters 1--9 serve as a self-contained replacement for Goodfellow et al.
  Chapters 5--8 if you prefer a single, more recent source.

**What to extract.** The intuition for *why* the identity path matters, in a form
you can reproduce at a whiteboard: unrolled, a residual network is a sum over
paths, and the gradient decomposes into a direct term plus a correction, so no
product of Jacobians can drive it to zero.

**Connection to the code.** `Add()([shortcut, x])` inside `res_block` and `rcab`,
and `Add(dtype='float32')([inp, x])` at the model's output, are Chapter 11
material. `Multiply()([x, s])` inside `se_block` is Chapter 12 material in
miniature.

**Relationship to the cited papers.** The textbook treatment behind He et al.
(2016), He et al. (2016b), Vaswani et al. (2017), and Ho et al. (2020).

## Wang & Bovik, *Modern Image Quality Assessment*

| Field | Value |
|---|---|
| Authors | Zhou Wang, Alan C. Bovik |
| Series | Synthesis Lectures on Image, Video, and Multimedia Processing |
| Publisher | Morgan & Claypool |
| Year | 2006 |
| ISBN | 978-1-59829-022-6 |
| Pages | ~108 |
| Level | Graduate; short and self-contained |
| Classification | **Core** |

**Why it is relevant to this project.** The enhanced notebook's loss function is
$0.8\,(1 - \mathrm{SSIM}) + 0.2\,\ell_1$. SSIM is therefore not a metric this
project merely reports --- it is the objective the network is trained to optimize,
which makes understanding its construction, its invariances, and its failure
modes a prerequisite rather than a nicety. This ~100-page monograph is written by
SSIM's inventors and covers the material more completely than the original 2004
paper.

**Topics it covers that appear in the technical document.**

* The critique of MSE and PSNR as perceptual measures --- the argument summarized
  in the technical document's Section 3.5.
* The structural-similarity philosophy: separating luminance, contrast, and
  structure, and the axioms each comparison function must satisfy (Section 3.6.1).
* The full derivation of the SSIM index, including why $C_3 = C_2/2$ collapses
  the three-factor product into the familiar closed form.
* Multi-scale SSIM (MS-SSIM), which is what Zhao et al. (2017) actually used in
  the loss mixture the notebook approximates with single-scale SSIM.
* The windowing question --- Gaussian versus uniform --- which is exactly the
  implementation discrepancy the technical document flags in Section 3.6.3,
  where the training loss uses TensorFlow's $11\times11$ Gaussian window and the
  reported metric uses scikit-image's $7\times7$ uniform default.

**Chapters to study.** The book is short enough to read end to end in a sitting
or two, and that is the recommendation. If time is very limited, read the
chapter developing the SSIM index and the chapter on multi-scale extension, and
skip the information-theoretic metrics (VIF and relatives) on a first pass.

**What to extract.** Two specifics. (i) That the structure term is a Pearson
correlation coefficient, hence invariant to affine intensity changes --- which is
why an SSIM-only loss can let absolute brightness drift, and therefore why the
$\ell_1$ anchor term in the notebook's `combined_loss` is not optional.
(ii) That SSIM is a *local* index averaged over windows, so the choice of window
changes the number; this is what makes the technical document's Section 3.6.3
warning material rather than pedantic.

**Connection to the code.** `tf.image.ssim(y_true, y_pred, max_val=1.0)` in
`combined_loss` and `skimage.metrics.structural_similarity` in the `SSIM` helper
are two different instantiations of this book's index. Reading it is what lets
you notice that they are not the same thing.

**Relationship to the cited papers.** The extended treatment of Wang et al.
(2004) and Wang et al. (2003), and essential background for Zhao et al. (2017),
whose loss function this project adapts.

## Chan & Shen, *Image Processing and Analysis*

| Field | Value |
|---|---|
| Authors | Tony F. Chan, Jianhong (Jackie) Shen |
| Subtitle | Variational, PDE, Wavelet, and Stochastic Methods |
| Publisher | Society for Industrial and Applied Mathematics (SIAM) |
| Year | 2005 |
| ISBN | 978-0-89871-589-7 |
| Pages | xxii + 400 |
| Level | Graduate, mathematically serious |
| Classification | **Core** |

**Why it is relevant to this project.** The technical document's Section 4
presents four classical families --- diffusion, total variation, wavelet
shrinkage, and non-local patch methods --- as a sequence of increasingly
sophisticated priors. Chan & Shen is the book that presents them as *one subject*
rather than four, unified by the variational formulation
$\min_u \tfrac{1}{2}\|u - y\|^2 + \lambda R(u)$ that the technical document uses
as its organizing equation. Its denoising chapter treats linear filtering, TV,
and wavelet methods under a single roof, which is exactly the perspective that
makes the technical document's regularizer table (Section 2.2) meaningful rather
than a list.

It is also the direct source for the ROF model's proper mathematical setting:
functions of bounded variation, why the $\ell_1$ gradient penalty admits jump
discontinuities while a quadratic penalty does not, and what "staircasing" is.

**Topics it covers that appear in the technical document.**

* Variational image models and the Bayesian/MAP interpretation of regularization
  (Sections 2.2 and 4.3).
* Total variation denoising: the ROF model, its Euler--Lagrange equation, the
  role of curvature motion, and the staircasing artifact (Section 4.3).
* PDE methods, including the heat equation as Gaussian smoothing and nonlinear
  diffusion (Sections 4.1--4.2).
* Wavelet-domain methods and their relationship to variational ones ---
  specifically, that soft thresholding is the proximal operator of an $\ell_1$
  penalty, which is the connection the technical document draws in Section 4.4.
* Stochastic and MRF image models, which is the probabilistic language in which
  "learned prior" eventually makes sense.

**Chapters to study.**

* **Chapter 2, *Some Modern Image Analysis Tools*** --- the mathematical
  toolkit: BV functions, wavelets, and the calculus of variations as used here.
  Read the BV material carefully; it is what makes the TV-preserves-edges claim
  precise.
* **Chapter 3, *Image Modeling and Representation*** --- how priors are
  constructed. Read for the conceptual frame.
* **Chapter 4, *Image Denoising*** --- **the essential chapter.** Read in full
  and in parallel with the technical document's Section 4.
* Chapters 5--7 (deblurring, inpainting, segmentation) --- read only if those
  problems interest you; deblurring is the natural next inverse problem after
  denoising and is a good demonstration of why a denoiser is a reusable prior
  (technical document Section 12.6).

**What to extract.** The unified variational viewpoint, and specifically the
understanding that choosing a denoising algorithm *is* choosing a prior. Once
that lands, the deep-learning move --- amortize the prior into network weights
instead of writing it down --- reads as a natural continuation of the same
programme rather than as a break with it.

**Connection to the code.** Indirect but deep: this book explains what the
network's weights have implicitly learned. It is also the theoretical companion
to Kok & Tam's Chapter 5.

**Relationship to the cited papers.** The textbook treatment of Rudin, Osher &
Fatemi (1992) and Perona & Malik (1990), and background for Donoho & Johnstone
(1994).

## Mallat, *A Wavelet Tour of Signal Processing: The Sparse Way*

| Field | Value |
|---|---|
| Author | Stéphane Mallat |
| Edition | 3rd edition |
| Publisher | Academic Press (Elsevier) |
| Year | 2008 |
| ISBN | 978-0-12-374370-1 |
| Pages | 832 |
| Level | Graduate to research |
| Classification | **Core** (for the denoising chapter) |

**Why it is relevant to this project.** The technical document's Section 4.4
makes three claims about wavelet shrinkage that it does not prove: that an
orthonormal transform preserves white Gaussian noise; that the universal
threshold $\sigma\sqrt{2\log N}$ is the right order of magnitude because the
maximum of $N$ standard normals concentrates near $\sqrt{2\log N}$; and that the
resulting estimator is within a logarithmic factor of an oracle. Mallat proves
all three. This is also the canonical source for the sparsity principle that
underlies BM3D's collaborative filtering.

**Topics it covers that appear in the technical document.**

* Orthonormal bases and the invariance of white noise under orthogonal transforms
  (Section 4.4).
* Nonlinear approximation and the relationship between sparsity and
  approximation error --- the reason a piecewise-smooth image has few large
  wavelet coefficients.
* Thresholding estimators: hard versus soft, risk analysis, and the oracle
  comparison.
* Minimax optimality over smoothness classes, i.e. the precise sense in which
  Donoho & Johnstone's result is a *guarantee*.
* Translation-invariant denoising / cycle spinning, which explains the
  pseudo-Gibbs ringing the technical document mentions.

**Chapters to study.**

* **Chapter 9, *Approximations in Bases*** --- read the nonlinear-approximation
  sections. This is the mathematical content of "images are sparse in a wavelet
  basis".
* **Chapter 11, *Denoising*** --- **the essential chapter.** Diagonal estimation
  in a basis, thresholding risk, the universal threshold, minimax optimality, and
  translation-invariant variants.
* Chapters 6--7 (wavelet zoom, wavelet bases) --- read as needed to support
  Chapters 9 and 11; the singularity/Lipschitz-regularity material in Chapter 6
  explains *why* edges produce large coefficients across scales.
* Chapters 1--5 --- reference material; do not read linearly.
* Skip on a first pass: compression, audio processing, and the geometric
  representation chapters.

**What to extract.** The oracle argument. Wavelet shrinkage works because there
exists an ideal estimator that knows which coefficients carry signal, and
thresholding approximates it to within a $\log N$ factor without that knowledge.
This is the classical era's cleanest statement of what a good prior buys you, and
it is the right yardstick against which to think about what a CNN's learned prior
buys instead.

**Connection to the code.** No direct correspondence --- the notebooks contain no
wavelets. The connection is conceptual and important: a convolutional network's
first layer is a learned filter bank, and the technical document's argument that
the network learns to separate signal from noise by *magnitude in a learned
representation* is the same argument, with the basis learned rather than
designed.

**Relationship to the cited papers.** The definitive treatment of Donoho &
Johnstone (1994), Donoho (1995), and Mallat (1989); essential background for
Dabov et al. (2007), whose 3-D collaborative filtering is a sparsity argument in
a data-adaptive grouping.

# Tier 3 --- Supplementary

Useful, but each addresses a specific gap rather than the main line.

## Hansen, *Discrete Inverse Problems: Insight and Algorithms*

| Field | Value |
|---|---|
| Author | Per Christian Hansen |
| Series | Fundamentals of Algorithms |
| Publisher | SIAM |
| Year | 2010 |
| ISBN | 978-0-89871-696-2 |
| Level | Graduate, computational |
| Classification | **Supplementary** |

**Why it is relevant to this project.** The technical document asserts in
Section 2.4 that denoising is ill-posed and that regularization is what makes it
determinate. Hansen is the book that makes "ill-posed" quantitative: singular
value decay, the discrete Picard condition, noise amplification along
small-singular-value directions, and the trade-off curve between data fidelity
and solution regularity. Denoising is the easiest possible instance (the forward
operator is the identity), so Hansen's machinery is more than the project needs
--- which is exactly why it is clarifying, because it shows what the *general*
problem looks like and hence what denoising is a special case of.

**Topics it covers that appear in the technical document.**

* Ill-posedness in Hadamard's sense and its discrete manifestation (Section 2.4).
* Tikhonov regularization, which is the $R(x) = \lambda\|\nabla x\|_2^2$ row of
  the technical document's regularizer table.
* Choosing the regularization parameter --- the L-curve and generalized
  cross-validation. This is the classical analogue of the question the enhanced
  notebook answers by training blind: how strong should the prior be, given that
  the right answer depends on the noise level?

**Chapters to study.** Chapters 1--3 for the setup and the SVD analysis of
ill-posedness; Chapter 4 for regularization methods; Chapter 5 for
parameter-choice methods. The iterative-regularization material later in the book
is optional here.

**What to extract.** The regularization-parameter-selection problem, and the
recognition that a *blind* denoiser must solve an amortized version of it: it has
to infer the appropriate strength from the input, which is precisely the argument
in the technical document's Section 8.2 for why blind training forces the network
to estimate $\sigma$ internally.

**Relationship to the cited papers.** Background for the general inverse-problem
framing that Venkatakrishnan et al. (2013) exploit in Plug-and-Play priors, and
context for Levin & Nadler (2011).

## Boyd & Vandenberghe, *Convex Optimization*

| Field | Value |
|---|---|
| Authors | Stephen Boyd, Lieven Vandenberghe |
| Publisher | Cambridge University Press |
| Year | 2004 |
| ISBN | 978-0-521-83378-3 |
| Free online | `https://stanford.edu/~boyd/cvxbook/` |
| Level | Graduate |
| Classification | **Supplementary** |

**Why it is relevant to this project.** Three specific results in the technical
document are convex-analysis facts. First, that soft thresholding is the proximal
operator of the $\ell_1$ norm (Section 4.4). Second, that clipping to $[0,1]$ is
Euclidean projection onto a convex set, and that such projections are
non-expansive --- which is the technical document's rigorous explanation for why
the measured noisy-input PSNR ($21.42$ dB) exceeds the theoretical $20.92$ dB,
and why `np.clip` on predictions can only help (Section 3.5). Third, that the MAP
objective is convex for the classical regularizers and hence has a unique
minimizer, which is what makes those methods well-defined.

**Chapters to study.** Chapter 2 (convex sets --- including projection onto a
set), Chapter 3 (convex functions), Chapter 4 (convex optimization problems), and
Section 8.1 for the projection material specifically. Chapter 5 (duality) is
worth reading if you intend to work with TV solvers, where the dual formulation is
how the problem is actually solved. The interior-point chapters are not needed
for this project.

**What to extract.** Proximal operators and projections as *objects*: once
$\operatorname{prox}$ and $P_C$ are familiar, a large fraction of classical
image restoration collapses into a small number of recognizable moves, and the
Plug-and-Play idea (replace a proximal operator with a denoiser) becomes an
obvious thing to try rather than a clever trick.

**Connection to the code.** `np.clip(..., 0.0, 1.0)` in `add_gaussian_noise` and
in every evaluation cell is $P_{[0,1]^N}$.

## Szeliski, *Computer Vision: Algorithms and Applications*

| Field | Value |
|---|---|
| Author | Richard Szeliski |
| Edition | 2nd edition |
| Series | Texts in Computer Science |
| Publisher | Springer |
| Year | 2022 |
| ISBN | 978-3-030-34371-2 |
| DOI | 10.1007/978-3-030-34372-9 |
| Pages | 947 |
| Free draft | `https://szeliski.org/Book/` |
| Level | Advanced undergraduate to graduate |
| Classification | **Supplementary** |

**Why it is relevant to this project.** Szeliski is the book that puts denoising
in context: it is one operator inside an imaging pipeline, and the choices made
upstream (sensor, demosaicing, colour transform, compression) determine what the
noise actually looks like by the time a denoiser sees it. That is precisely the
issue behind the technical document's Sections 9.2 and 11.6 --- LFW's "clean"
targets are JPEG-compressed photographs that already contain real sensor noise
and demosaicing artifacts, so the experiment measures inversion of synthetic AWGN
applied on top of an imperfect reference. The 2nd edition also has a
deep-learning chapter, so it bridges the classical and modern halves of the
technical document in a single volume.

**Chapters to study.** The image-formation chapter for the sensor pipeline and
photon noise; the image-processing chapter for linear and non-linear filtering
(including bilateral filtering, the missing link between Gaussian smoothing and
non-local means in the technical document's Section 4); the model-fitting and
optimization chapter for regularization, MRFs and variational methods; and the
deep-learning chapter for a compact modern survey. The computational-photography
material covers practical denoising and multi-image fusion.

**What to extract.** The end-to-end pipeline picture, and the bilateral filter,
which the technical document mentions only in passing but which is conceptually
the halfway house between local averaging and NLM: it weights neighbours by both
spatial *and* intensity proximity, making it the first widely used
content-adaptive averaging scheme.

**Relationship to the cited papers.** Background for Tomasi & Manduchi (1998),
and general context for the real-noise citations, Plotz & Roth (2017) and Guo et
al. (2019).

# Tier 4 --- Advanced and Specialist

Read these when the Core tier is comfortable, or consult them as references.

## Elad, *Sparse and Redundant Representations*

| Field | Value |
|---|---|
| Author | Michael Elad |
| Subtitle | From Theory to Applications in Signal and Image Processing |
| Publisher | Springer |
| Year | 2010 |
| ISBN | 978-1-4419-7010-7 |
| DOI | 10.1007/978-1-4419-7011-4 |
| Level | Graduate to research |
| Classification | **Advanced** |

**Why it is relevant to this project.** This is the definitive treatment of the
sparse-representation view of denoising: model a patch as $x \approx D\alpha$
with $\alpha$ sparse, and denoise by finding the sparsest representation
consistent with the data. It is the row of the technical document's regularizer
table (Section 2.2) labelled "Sparse coding / K-SVD", and it is the intellectual
context in which BM3D was invented. Crucially for this project, it is also the
clearest statement of the *learned dictionary* idea --- the first move from
hand-designed to data-adapted priors, and therefore the direct conceptual
predecessor of learning the prior with a CNN.

**Topics it covers that appear in the technical document.** Sparsity as a prior;
$\ell_0$ versus $\ell_1$ and why the convex relaxation is justified; pursuit
algorithms (matching pursuit, basis pursuit); dictionary learning via K-SVD; and
patch-based image denoising built on all of the above.

**Chapters to study.** The theoretical first part (uniqueness of sparse
solutions, pursuit algorithms and their guarantees) can be read selectively ---
the results matter more than the proofs for this project's purposes. The
applications part is what to read carefully: the chapters on dictionary learning
and on image denoising, where the K-SVD denoising algorithm is developed. Read
the denoising application alongside the technical document's Section 4.6.

**What to extract.** The patch-modelling paradigm, and the recognition that
*every* strong classical denoiser --- NLM, BM3D, K-SVD --- is a patch method. The
technical document's Section 4.7 argues that the classical plateau came from the
difficulty of extending patch size; Elad's book is where you see why patch size is
the binding constraint.

**Relationship to the cited papers.** The book-length treatment of Elad & Aharon
(2006), and the natural companion to Dabov et al. (2007).

## Aubert & Kornprobst, *Mathematical Problems in Image Processing*

| Field | Value |
|---|---|
| Authors | Gilles Aubert, Pierre Kornprobst |
| Subtitle | Partial Differential Equations and the Calculus of Variations |
| Edition | 2nd edition |
| Series | Applied Mathematical Sciences, vol. 147 |
| Publisher | Springer |
| Year | 2006 |
| ISBN | 978-0-387-32200-1 |
| DOI | 10.1007/978-0-387-44588-5 |
| Level | Research; assumes real analysis and functional analysis |
| Classification | **Advanced** |

**Why it is relevant to this project.** The technical document states in
Section 4.2 that the Perona--Malik equation is *ill-posed* because its diffusion
coefficient across strong edges, $g(s) + s\,g'(s)$, becomes negative --- running
the heat equation backwards. It states in Section 4.3 that the ROF model is
well-posed in the space of functions of bounded variation. Both are genuine
theorems, and this is the book that proves them. Read it if you want the
variational and PDE sections of the technical document on a rigorous footing
rather than an intuitive one.

**Topics it covers that appear in the technical document.** Existence and
uniqueness for the ROF minimization problem in BV; the mathematical pathologies
of the Perona--Malik model and the regularized (Catté--Lions--Morel) variants that
repair them; viscosity solutions; and the general calculus-of-variations
machinery (direct method, lower semicontinuity, relaxation) that image models
require.

**Chapters to study.** The mathematical-preliminaries chapter as a reference for
BV, Sobolev spaces, and the direct method; then the image-restoration chapter,
which is where the denoising and deblurring models are analysed. The edge
detection, segmentation and sequence-analysis chapters are outside this project's
scope.

**What to extract.** Precision about a claim the technical document makes
informally: that TV "preserves edges" is not a heuristic but a consequence of the
fact that BV functions admit jump discontinuities while $H^1$ functions do not.

**Relationship to the cited papers.** The rigorous treatment behind Perona &
Malik (1990) and Rudin, Osher & Fatemi (1992).

## Barbu, *Novel Diffusion-Based Models for Image Restoration and Interpolation*

| Field | Value |
|---|---|
| Author | Tudor Barbu |
| Series | Signals and Communication Technology |
| Publisher | Springer |
| Year | 2019 (hardcover 2018) |
| ISBN | 978-3-319-93005-3 (hardcover); 978-3-030-06566-9 (softcover) |
| DOI | 10.1007/978-3-319-93006-0 |
| Level | Research monograph |
| Classification | **Advanced / specialist** |

**Why it is relevant to this project.** A focused monograph on nonlinear
diffusion for restoration --- the family the technical document introduces in
Section 4.2 through Perona & Malik. Where Aubert & Kornprobst give the classical
analysis, Barbu surveys and develops *modern variants*: second- and fourth-order
anisotropic diffusion models, hybrid schemes, and their numerical
discretizations, with restoration and interpolation as the driving applications.

**Why it is worth reading despite the deep-learning turn.** Two reasons specific
to this project. First, the technical document notes that Chen & Pock's (2017)
trainable nonlinear reaction diffusion unrolls exactly these PDE models into a
learnable feed-forward network --- so understanding the diffusion models is
understanding what that architecture is a learned version of. Second, "diffusion"
in the modern generative sense (Ho et al., 2020) is a *different* object that
shares a name; reading a book on classical image diffusion is a good inoculation
against conflating the two, a confusion that is common and unhelpful.

**Chapters to study.** Read selectively: the chapters developing the nonlinear
diffusion denoising models and their numerical schemes are the relevant ones; the
interpolation and inpainting material is adjacent rather than central here. Treat
it as a survey to sample rather than a textbook to work through.

**What to extract.** A sense of how far the hand-designed-PDE programme was
pushed, and its performance ceiling --- which is the empirical backdrop to the
technical document's Section 4.7 argument about why the field moved to learned
priors.

**Relationship to the cited papers.** Extends Perona & Malik (1990); provides the
model class that Chen & Pock (2017) learn.

# Coverage Matrix: Technical Document to Books

The following table maps each substantive section of the technical document to
the book chapters that cover it. Use it to read the two documents together.

| Technical-doc section | Topic | Primary source | Secondary |
|---|---|---|---|
| 1.2 | Sensor noise, Poisson--Gaussian | Gonzalez & Woods Ch. 5 | Szeliski (image formation) |
| 2.1 | AWGN observation model | Kok & Tam Ch. 1 | Gonzalez & Woods Ch. 5 |
| 2.2 | MAP estimation, regularizer table | Kay Ch. 10--11 | Chan & Shen Ch. 3--4 |
| 2.3 | MMSE, Tweedie's formula | Kay Ch. 11 | Prince Ch. 18 |
| 2.4 | Ill-posedness | Hansen Ch. 1--3 | Aubert & Kornprobst (prelims) |
| 3.1 | Convolution, equivariance | Goodfellow Ch. 9 | Gonzalez & Woods Ch. 3 |
| 3.2 | Receptive field | Prince Ch. 10 | Goodfellow Ch. 9 |
| 3.3 | Residual learning, gradient flow | **Prince Ch. 11** | Goodfellow Ch. 8 |
| 3.4 | Batch normalization | Prince Ch. 11 | Goodfellow Ch. 8 |
| 3.5 | PSNR; clipping as projection | Wang & Bovik | Boyd & Vandenberghe Ch. 2, §8.1 |
| 3.6 | SSIM derivation | **Wang & Bovik** | Kok & Tam Ch. 1 |
| 4.1 | Spatial filtering | Gonzalez & Woods Ch. 3, 5 | Kok & Tam Ch. 2 |
| 4.2 | Anisotropic diffusion | Chan & Shen Ch. 4 | Aubert & Kornprobst; Barbu |
| 4.3 | Total variation / ROF | **Chan & Shen Ch. 4** | Kok & Tam Ch. 5; Aubert & Kornprobst |
| 4.4 | Wavelet shrinkage | **Mallat Ch. 11** | Kok & Tam Ch. 3 |
| 4.5 | Non-local means | Kok & Tam Ch. 6 | Szeliski (filtering) |
| 4.6--4.7 | BM3D; why classical plateaued | Elad (applications) | Kok & Tam Ch. 4 |
| 5 | Research evolution | Prince Ch. 10--12 | Szeliski (deep learning) |
| 6 | RCAB, Squeeze-and-Excitation | Prince Ch. 11--12 | Goodfellow Ch. 9 |
| 7.1--7.4 | MSE blur, $\ell_1$, SSIM loss | Kay Ch. 11 + Wang & Bovik | Goodfellow Ch. 5 |
| 7.5--7.6 | Cosine schedule, Adam, mixed precision | Goodfellow Ch. 8 | Prince Ch. 6--7 |
| 8 | Patches, blind $\sigma$, augmentation | Goodfellow Ch. 7 | Prince Ch. 9 |
| 9 | Code walkthrough | (no book; read the notebooks) | Kok & Tam for classical analogues |
| 10 | Experiments and results | Wang & Bovik (metrics) | Kok & Tam Ch. 1 |
| 11 | Limitations | Hansen Ch. 5 (parameter choice) | Szeliski (real pipelines) |
| 12 | Future directions | Prince Ch. 18 (diffusion) | Elad (sparsity), Szeliski |

## Where the books run out

Three parts of the technical document have **no adequate book coverage**, and
must be learned from the papers themselves:

1. **Channel attention specifically** (Sections 6.3--6.4). Prince's transformer
   chapter gives the general attention pattern, but Squeeze-and-Excitation and
   RCAB are covered only in Hu et al. (2018) and Zhang et al. (2018). Both papers
   are short and readable; read them directly.
2. **The modern restoration state of the art** (Section 5.6). NAFNet (Chen et
   al., 2022) and Restormer (Zamir et al., 2022) postdate every book here except
   Prince and Szeliski, neither of which covers them. Read the papers.
3. **Self-supervised denoising** (Section 12.3). Noise2Noise, Noise2Void, and
   Neighbor2Neighbor have no textbook treatment at all. The Noise2Noise argument
   is a two-line consequence of Kay's Chapter 11, however: if
   $\mathbb{E}[y' \mid x] = x$, then regressing onto $y'$ has the same minimizer
   as regressing onto $x$.

# Suggested Study Plans

## Plan A --- Six weeks, code-focused

| Week | Reading | Deliverable |
|---|---|---|
| 1 | Gonzalez & Woods Ch. 2--3, 5 | Reimplement `add_gaussian_noise` and a mean/median filter from scratch; verify the $20.92$ dB prediction |
| 2 | Kok & Tam Ch. 1--3 | Run wavelet shrinkage on the LFW test set; record PSNR/SSIM |
| 3 | Goodfellow Ch. 7--9 | Annotate every layer of `build_model` with its parameter count; confirm $373{,}057$ |
| 4 | Prince Ch. 10--11 | Derive the receptive fields ($25$ and $37$) independently; verify by ablating blocks |
| 5 | Wang & Bovik (whole) | Reproduce both SSIM variants; quantify the Gaussian-vs-uniform window gap |
| 6 | Kok & Tam Ch. 5--6 | Add TV and NLM rows to the results table; write up |

## Plan B --- One semester, theory-focused

Weeks 1--2: Kay Ch. 10--12. Weeks 3--4: Gonzalez & Woods Ch. 3--5 and Kok & Tam
Ch. 1--3. Weeks 5--6: Chan & Shen Ch. 2--4. Weeks 7--8: Mallat Ch. 9, 11.
Weeks 9--10: Goodfellow Ch. 7--9. Weeks 11--12: Prince Ch. 10--12. Weeks 13--14:
Wang & Bovik, plus Hansen Ch. 4--5. Final weeks: run the ablation grid from the
technical document's Section 12.1 and write it up.

## Plan C --- Research preparation

Plan B, then: Elad (applications), Aubert & Kornprobst (restoration chapter),
Barbu (selectively), Prince Ch. 18, and the paper list in the technical
document's Sections 5.6 and 12. At that point the natural project is the one the
technical document identifies as most valuable --- a properly controlled ablation
separating the six confounded changes between the baseline and enhanced
notebooks, on a fixed seed and a leakage-free split.

# Summary Table

| Book | Year | ISBN | Tier | Essential chapters |
|---|---|---|---|---|
| Gonzalez & Woods, *Digital Image Processing*, 4th ed. | 2018 | 978-0-13-335672-4 | Foundational | 2, 3, 4, 5, 7 |
| Kay, *Estimation Theory*, Vol. I | 1993 | 978-0-13-345711-7 | Foundational | 10, 11, 12 |
| Kok & Tam, *Digital Image Denoising in MATLAB* | 2024 | 978-1-119-61769-3 | Core | 1--7 (all) |
| Goodfellow et al., *Deep Learning* | 2016 | 978-0-262-03561-3 | Core | 5, 6, 7, 8, 9 |
| Prince, *Understanding Deep Learning* | 2023 | 978-0-262-04864-4 | Core | 10, 11, 12, 18 |
| Wang & Bovik, *Modern Image Quality Assessment* | 2006 | 978-1-59829-022-6 | Core | all (~108 pp.) |
| Chan & Shen, *Image Processing and Analysis* | 2005 | 978-0-89871-589-7 | Core | 2, 3, 4 |
| Mallat, *A Wavelet Tour*, 3rd ed. | 2008 | 978-0-12-374370-1 | Core | 9, 11 |
| Hansen, *Discrete Inverse Problems* | 2010 | 978-0-89871-696-2 | Supplementary | 1--5 |
| Boyd & Vandenberghe, *Convex Optimization* | 2004 | 978-0-521-83378-3 | Supplementary | 2, 3, 4, §8.1 |
| Szeliski, *Computer Vision*, 2nd ed. | 2022 | 978-3-030-34371-2 | Supplementary | image formation, processing, optimization, deep learning |
| Elad, *Sparse and Redundant Representations* | 2010 | 978-1-4419-7010-7 | Advanced | applications part |
| Aubert & Kornprobst, *Mathematical Problems in Image Processing*, 2nd ed. | 2006 | 978-0-387-32200-1 | Advanced | preliminaries, restoration |
| Barbu, *Novel Diffusion-Based Models* | 2019 | 978-3-319-93005-3 | Advanced | diffusion denoising models |

**Fourteen books.** Four of them (Kok & Tam, Goodfellow et al., Prince, Wang &
Bovik) are sufficient to understand and extend the repository. The remaining ten
are what turn that understanding into the ability to judge whether the approach
was the right one.
