# FetMRQC SR

FetMRQC SR is the super-resolution extension of FetMRQC [[paper1](https://arxiv.org/pdf/2304.05879.pdf),[paper2](https://arxiv.org/pdf/2311.04780.pdf)] is a tool for quality assessment (QA) and quality control (QC) of T2-weighted (T2w) fetal brain MR images. 

It builds on top of the utilities developed in the [FetMRQC repository](https://github.com/Medical-Image-Analysis-Laboratory/fetmrqc), a tool for the QC of low-resolution T2w scans.

It contains the tools needed

It consists of two parts.
1. A **rating interface** (visual report) to standardize and facilitate quality annotations of T2w fetal brain MRI images, by creating interactive HTML-based visual reports from fetal brain scans. It uses a pair of low-resolution (LR) T2w images with corresponding brain masks to provide snapshots of the brain in the three orientations of the acquisition in the subject-space. 
2. A **QA/QC model** that can predict the quality of given super-resolution reconstructed volumes. 

This is a work in progress. Please look at the original [FetMRQC repository](https://github.com/Medical-Image-Analysis-Laboratory/fetmrqc) for the T2w QC tool

If you have used this work in your research, please cite:

https://www.arxiv.org/abs/2503.10156

## License
Copyright 2025 Medical Image Analysis Laboratory. 

## Acknowledgements
This project was supported by the ERA-net Neuron MULTIFACT – SNSF grant [31NE30_203977](https://data.snf.ch/grants/grant/203977).

