<div align="center">

<samp>

<h1> WaveLightNet: A Wavelet Decomposition Filter based CNN-LSTM Network for 6DOF Pose Estimation of Origami Robot  </h1>

<h3> Adithya K Krishna, Seenivasan lalithkumar, Hongliang Ren </h3>

</samp>   

</div>     

### Wavelet Decomposition
<p>
  <img src="https://github.com/AdithyaK243/WaveLightNet/blob/main/figures/Wavelet%20Decomposition.png" width=300 height=300 >  
</p>

### Proposed Architecture
<p>
  <img src="https://github.com/AdithyaK243/WaveLightNet/blob/main/figures/Model%20Architecture.png" width=900 height=500 >  
</p>

## Datasets
Download the datasets from these repositories and place them under the main repository, path of Data and Linemod are added by default and can be used for training

 Base Dataset(Used in SOTA & Referred in paper) </br>
   - https://gitlab.com/ruphan/origami-worm-pose-dataset </br>
   
 LINEMOD: </br> 
   - https://cvarlab.icg.tugraz.at/projects/3d_object_detection/ObjRecPoseEst.tar.gz </br>
   - https://github.com/AdithyaK243/Linemod </br>
   
 Dataset(used in paper)</br>  
  - https://github.com/AdithyaK243/Data


## Directory setup
<!---------------------------------------------------------------------------------------------------------------->
The structure of the repository is as follows: 

- `dataset/`: Contains the data needed to train the network.
- `checkpoints/`: Contains trained weights for WavelightNet, ablation study, linemod .
- `models/`: Contains base CNN-LSTM network and Wavelet Feature Extraction codes.
- `utils/`: Contains utility tools used noise and occlusion study.
- `loader`: Data loader for training and testing for data used in paper.
- `main` : Main python file to run training and testing.

---

## Dependencies
- Python 3.7
- Tensorflow (2.x)
- PyWavelet (Wavelet Decomposition)

### Training and Testing 
Arguments

- lr
- epoch 
- batch_size
- case : Boolean value to decide whether to add or remove Wavelet feature extraction (default:True)
- shuffle_data : Shuffle data while loading  (default :False)
- shuffle_train: Shuffle while forming batches (default :False)
- test_model: Whether to carry out training or testing (default: False)
- model_name : ['Cnn', 'CnnLstm', 'Linemod'] one of the three names can be used (default: CnnLstm)
- chkpt_path: Path to checkpoint file when test_model is True    

For Training:
```bash
python main.py --case --True --test_model False
```
For Testing:
```bash
python main.py --test_model True chkpt_path <path to h5 saved weights>
```

## Results
 Comparison with SOTA:

| 6-D | ScoopNet | Ours |
|:-:|:-:|:-:|
Pose Vecor | Mean(cm)-Mean Dev | Mean(cm)-Mean Dev| 
Tx | 0.0018 - 0.0007 | **0.0004** - **0.0001** 
Ty | 0.0017 - 0.0006 | **0.0006** - **0.0002** 
Tz | **0.0607** - 0.0038 | 0.0738 - **0.0117** 
Rx | 4.8255 - 0.7527 | **4.4678** - **0.1163** 
Ry | 0.4279 - 0.0233 | **0.2963** - **0.1050** 
Rz | 0.5173 - 0.1706 | **0.0606** - **0.0039** 


## Contact

For any queries, please contact [Adithya K Krishna](mailto:adithya.krishnakumar@gmail.com) or [Lalithkumar Seenivasan](mailto:lalithjets@gmail.com)