# Third-party presentation assets

This ENCCS training site uses the MIT-licensed CodeRefinery `sphinx-lesson`
extension and the MIT-licensed ENCCS `sphinx-evita` extension. It uses Furo
for HTML presentation and MyST-NB for the Markdown notebook source. Their
versions are declared in `requirements.txt`.

The ENCCS light/dark logos and favicon in `_static/` are copied from the
[ENCCS course template](https://github.com/ENCCS/python-for-hpc),
which provides a [CC BY-SA 4.0 content license](https://github.com/ENCCS/python-for-hpc/blob/main/LICENSE)
and [MIT code license](https://github.com/ENCCS/python-for-hpc/blob/main/LICENSE.code). The assets
are included for this ENCCS course; no course prose or scientific code was
copied. The MLIP lesson prose and example code here are original project
content.

The site setup pages also include two resized photographs licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):

- `_static/jupiter-booster-racks.jpg`: JUPITER Booster racks by
  Forschungszentrum Jülich / Sascha Kreklau;
  [Wikimedia Commons source](https://commons.wikimedia.org/wiki/File:JUPITER_racks_with_logos_of_supporters_and_partners._Copyright-_Forschungszentrum_J%C3%BClich_-_Sascha_Kreklau.jpg).
- `_static/leonardo-cabinets.png`: Leonardo supercomputer by the National
  Institute of Geophysics and Volcanology (INGV);
  [Wikimedia Commons source](https://commons.wikimedia.org/wiki/File:Leonardo_supercomputer.png).

Only the image dimensions were changed. Their inclusion does not imply that
the photographers or institutions endorse this lesson.

## Runtime dependencies (not redistributed)

The Part A pages install these packages and download these model weights at
run time. Nothing from them is copied into this repository.

- [TorchSim](https://github.com/TorchSim/torch-sim) (`torch-sim-atomistic`): MIT.
- [mace-torch](https://github.com/ACEsuit/mace) and the MACE-MP-0b checkpoint: MIT.
- [orb-models](https://github.com/orbital-materials/orb-models) and the Orb-v3
  checkpoints: Apache-2.0.

## MatGL tutorials (adapted workflows)

The scripts in `examples/matgl/` and `examples/training/` follow the workflows of the
[MatGL tutorials](https://github.com/materialsvirtuallab/matgl/tree/main/examples)
(relaxation and MD with a universal potential, lattice-constant benchmark,
relax-then-predict with property models, formation-energy training and potential
fine-tuning). MatGL and its
tutorials are distributed under the
[BSD 3-Clause licence](https://github.com/materialsvirtuallab/matgl/blob/main/LICENSE),
Copyright (c) Materials Virtual Lab. The code here was rewritten as small
modules with different systems, a generated Cu dataset and a float64 switch; the
pretrained MatGL weights are downloaded at run time and not redistributed.
