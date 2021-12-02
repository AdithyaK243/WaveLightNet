# WaveLightNet: A Wavelet Decomposition Filter based CNN-LSTMNetwork for 6DOF Pose Estimation of Origami Robot

### Wavelet Decomposition
<p>
  <img src="https://github.com/AdithyaK243/WaveLightNet/blob/main/figures/Wavelet%20Decomposition.png" width=300 height=300 >  
</p>

### Proposed Architecture
<p>
  <img src="https://github.com/AdithyaK243/WaveLightNet/blob/main/figures/Model%20Architecture.png" width=900 height=600 >  
</p>

## Dataset
 Base Dataset </br>
   - https://gitlab.com/ruphan/origami-worm-pose-dataset </br>
   
 LINEMOD: </br> 
   - https://cvarlab.icg.tugraz.at/projects/3d_object_detection/ObjRecPoseEst.tar.gz </br>
   - https://github.com/AdithyaK243/Linemod </br>
   
 Dataset </br>  
  - https://github.com/AdithyaK243/Data

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

