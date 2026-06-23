# GAD-MambaUNet BIBM Package

This folder is a self-contained working package for the BIBM manuscript and its accompanying source code.

## Layout

- `paper/`: IEEE conference manuscript, bibliography, IEEEtran class file, architecture figure, and compiled PDF.
- `code/gt-mamba-unet/`: GAD-MambaUNet and MK-UNet implementations, training scripts, DINOv3 teacher wrapper, and selective-scan CUDA source.

## Compile the manuscript

From `paper/`, run:

```bash
latexmk -pdf IEEE-conference-template-062824.tex
```

The paper depends only on the files included in `paper/`, including `IEEEtran.cls`, `paper_refs.bib`, and `all-network.pdf`.

## Code notes

The code package includes source files only. Python cache files, generated CUDA build artifacts, and package metadata were intentionally omitted because they are platform- and Python-version-specific and can be regenerated locally. Datasets, pretrained DINOv3 weights, trained checkpoints, and environment dependencies are not included in this package.

The main proposed model is `code/gt-mamba-unet/networks/Asym_GroupedMamba_UNet_DSI_GAM_v3.py`. The baseline implementation is `code/gt-mamba-unet/networks/mkunet_network.py`.
