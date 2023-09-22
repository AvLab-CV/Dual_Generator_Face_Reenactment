
# Dual-Generator-Face-Reenactment-DG

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![CUDA 10.2](https://img.shields.io/badge/cuda-10.2-green.svg?style=plastic)
![Pytorch 1.6](https://img.shields.io/badge/pytorch-1.60-green.svg?style=plastic)

![image](https://github.com/AvLab-CV/Dual_Generator_Face_Reenactment/blob/main/qrcode.png)
![image](https://github.com/AvLab-CV/Dual_Generator_Face_Reenactment/blob/main/result.gif)

![image](https://github.com/AvLab-CV/Dual_Generator_Face_Reenactment/blob/main/result2.gif)
![image](https://github.com/AvLab-CV/Dual_Generator_Face_Reenactment/blob/main/github_sample.png)
![image](https://github.com/AvLab-CV/Dual_Generator_Face_Reenactment/blob/main/github_sample2.png)

**Abstract:** We propose the Dual-Generator (DG) network for large-pose face reenactment. Given a source face and a reference face as inputs, the DG network can generate an output face that has the same pose and expression as of the reference face, and has the same identity as of the source face. As most approaches do not particularly consider large-pose reenactment, the proposed approach addresses this issue by incorporating a 3D landmark detector into the framework and considering a loss function to capture visible local shape variation across large pose. The DG network consists of two modules, the ID-preserving Shape Generator (IDSG) and the Reenacted Face Generator (RFG). The IDSG encodes the 3D landmarks of the reference face into a reference landmark code, and encodes the source face into a source face code. The reference landmark code and the source face code are concatenated and decoded to a set of target landmarks that exhibits the pose and expression of the reference face and preserves the identity of the source face. 

## Virtual environment
- Clone this repo to your desired folder
    ```
    git clone https://github.com/Charles8745/Dual_Generator_Face_Reenactment.git
    ```
- Move to Dual_Generator_Face_Reenactment folder
    ```
    cd Dual_Generator_Face_Reenactment
    ```
- Establish a virtual environment
    ```
    conda env create -f environment.yml
    conda activate reenactment
    pip install -r requirement.txt
    ```

## Download models
- Main model

    https://drive.google.com/file/d/1gHK7NObVP1c1fVsVGdZe6t21CmmtarVS/view?usp=sharing
    
    Unzip it and place the 'expr' and 'expr_lm' folders into the main directory.


    

- Vgg model

    https://drive.google.com/file/d/16b_84dT9wq-CEZrKeR5cvN39zGCC7np9/view?usp=sharing

    Place it into the main directory.

## Inference
- Run the demo.py
    ```
    python demo.py
    ```

## Details of implementataion

![image](https://github.com/AvLab-CV/Dual_Generator_Face_Reenactment/blob/main/IDSG.JPG)
![image](https://github.com/AvLab-CV/Dual_Generator_Face_Reenactment/blob/main/RFG.JPG)

