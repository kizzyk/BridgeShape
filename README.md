# BridgeShape: Latent Diffusion Schrödinger Bridge for 3D Shape Completion [AAAI 2026]
## [Paper](https://arxiv.org/abs/2506.23205)

Official implementation of the paper **"BridgeShape: Latent Diffusion Schrödinger Bridge for 3D Shape Completion"** (AAAI 2026).


---

## 📢 News & Updates
* **[March 2026]** Initial release of the core BridgeShape framework (Latent Diffusion Schrödinger Bridge). 
* **[Upcoming]** Release of the Depth-Enhanced VQ-VAE module featuring the self-projected Multi-View depth feature. Stay tuned!

---

## 🛠️ Environments

You can easily set up and activate a conda environment for this project by using the following commands:

```bash
conda env create -f environment.yml
conda activate bridgeshape
```

## 📂 Data Construction

We utilize the 3D-EPN dataset for our experiments. Please download the original data available from the [3D-EPN](https://graphics.stanford.edu/projects/cnncomplete/data.html) project page for both training and evaluation.

To run the default setting with a resolution of 32³, download the necessary data files [shapenet_dim32_df.zip](http://kaldir.vc.in.tum.de/adai/CNNComplete/shapenet_dim32_df.zip) (completed shapes) and [shapenet_dim32_sdf.zip](http://kaldir.vc.in.tum.de/adai/CNNComplete/shapenet_dim32_sdf.zip) (partial shapes).

To prepare the data:

Run ```data/sdf_2_npy.py``` to convert the raw files into ```.npy``` format for easier handling.

Run ```data/npy_2_pth.py``` to generate the paired data for the eight object classes used for model training.

Your data structure should be organized as follows before starting the training process:

```
BridgeShape
├── data
│   ├── 3d_epn
│   │   ├── 02691156
│   │   │   ├── 10155655850468db78d106ce0a280f87__0__.pth
│   │   │   ├── ...  
│   │   ├── 02933112
│   │   ├── 03001627
│   │   ├── ...
│   │   ├── splits
│   │   │   ├── train_02691156.txt
│   │   │   ├── train_02933112.txt
│   │   │   ├── ...  
│   │   │   ├── test_02691156.txt
│   │   │   ├── test_02933112.txt
│   │   │   ├── ...
```

## 🚀 Training
BridgeShape employs a two-stage training pipeline. We train category-specific models for the eight distinct categories.

### Stage 1: Train & Test VQ-VAE
First, train the VQ-VAE to establish the compact latent space. We provide shell scripts for each of the eight categories in the ```scripts/``` directory.

Simply run the corresponding ```.sh``` file for your desired category:
```angular2html
bash scripts/train_vqvae_snet_*.sh
```
Note: The script handles both training and testing. The best performing weights will be automatically saved to ```ckps/vqvae_epoch-best.pth```.

### Stage 2: Train Latent Diffusion Schrödinger Bridge
Once the VQ-VAE is trained, you can train the diffusion bridge model using the configuration files located in the ```configs/``` directory.
```angular2html
CUDA_VISIBLE_DEVICES=0 python train_vqsdf.py \
    --save_dir [YOUR_SAVE_DIRECTORY] \
    --config configs/[YOUR_CONFIG].yaml
```

## 🔍 Inference / Testing
To evaluate your trained BridgeShape model, run the testing script with your desired sampling configurations:
```angular2html
CUDA_VISIBLE_DEVICES=0 python test_vqsdf.py \
    --save_dir [YOUR_SAVE_DIRECTORY] \
    --config configs/[YOUR_CONFIG].yaml \
    --rs 1 \
    --tbs 10 \
    --test_start_epoch 5 \
    --test_one false \
    --v true
```

**Testing Arguments:**
* `--rs`: Number of reverse sampling steps.
* `--tbs`: Test batch size.
* `--test_start_epoch`: The epoch number from which to start testing.
* `--test_one`: Set to `true` to run only a single evaluation round.
* `--v`: Set to `true` to save the output visualization results.

## 📖 Citation
If you find our work or code useful for your research, please consider citing our paper:
```
@inproceedings{kong2026bridgeshape,
  title={BridgeShape: Latent Diffusion Schr{\"o}dinger Bridge for 3D Shape Completion},
  author={Kong, Dequan and Chen, Honghua and Zhu, Zhe and Wei, Mingqiang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={7},
  pages={5726--5734},
  year={2026}
}
```

## 🙏 Acknowledgement
Our implementation builds upon several excellent open-source projects. We express our gratitude to the authors of:

 [DiffComplete](https://github.com/JIA-Lab-research/DiffComplete) for structural inspiration.

 [P2P-Bridge](https://github.com/matvogel/P2P-Bridge) for insights into diffusion bridges.
