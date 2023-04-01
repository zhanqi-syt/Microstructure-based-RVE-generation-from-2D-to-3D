If useful to your research, we would appreciate a citation:<br>
***Yutai Su, Ziyi Shen, Xu Long, Chuantong Chen, Lehua Qi, Xujiang Chao. Gaussian filtering method of evaluating the elastic/elasto-plastic properties of sintered nanocomposites with quasi-continuous volume distribution[J]. Materials Science and Engineering: A, 2023, XXX.***<br>
Feel free to utilize this code. If any questions, please email us (suyutai@nwpu.edu.cn). <br>

# Microstructure-based-RVE-generation-from-2D-to-3D
## Keywords: 
Metal matrix nanocomposites; Gaussian random field; RVE; Quasi-continuous matrix; Elastic/elasto-plastic properties
## Introduction
Here is a consise flowchart to describe this work, as shown below:<br>
1. Pre-process the 2D or 3D real morphology and obtain the **volume fraction $\phi_0$** (binary statistics) and **two-point probability function $S_2$** (FFT, fast Fourier transformation).
2. Generate the initial 3D synthetic microstructure based on the target **volume fraction $\phi_0$** and the initial **Gaussian kernel $\sigma_0=1.0$**.
3. Obtain the synthetic **two-point probability function $S_2^{\prime}$** from 3D synthetic microstructure using FFT.
4. Establish a optimization problem to obtain the minimum value of mean absolute percentage error (MAPE) between the real $S_2$ and the synthetic $S_2^{\prime}$.
5. Obtain the optimized **Gaussian kernel $\sigma$** and the final 3D synthetic RVE microstructure.

Python codes are freely provided on GitHub for this process. (github.com/zhanqi-syt/Microstructure-based-RVE-generation-from-2D-to-3D)

![图7](https://user-images.githubusercontent.com/116877222/229271220-e0396908-f1a7-4ea1-bdfa-ebec7f7803ec.png)
<br>**Fig. 1. Flowchart of the synthetic RVE generation from real microstructures.**<br>
<br>
<br>
![图8](https://user-images.githubusercontent.com/116877222/229271211-55b1d327-ee08-445c-8a65-237e0ad30032.png)
<br>**Fig. 2. Microstructures and two-point probability functions of sintered metal particles: (a) real microstructure, (b) generated RVE, and (c) two-point probability functions of real microstructure and generated RVE.**<br>









