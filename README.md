If useful to your research, we would appreciate a citation:<br>
***Yutai Su, Ziyi Shen, Xu Long, Chuantong Chen, Lehua Qi, Xujiang Chao. Gaussian filtering method of evaluating the elastic/elasto-plastic properties of sintered nanocomposites with quasi-continuous volume distribution[J]. Materials Science and Engineering: A, 2023, XXX.***<br>
Feel free to utilize this code. If any questions, please email us (suyutai@nwpu.edu.cn). <br>

# Microstructure-based-RVE-generation-from-2D-to-3D
## Keywords: 
Metal matrix nanocomposites; Gaussian random field; RVE; Quasi-continuous matrix; Elastic/elasto-plastic properties
## Flowchart for Generating Synthetic Microstructure Based on Real Morphology
The key steps involved in generating a 3D synthetic microstructure that matches a given 2D or 3D real morphology are summarized below:<br>
1. **Pre-processing:** Calculate the volume fraction $\phi_0$ (obtained via binary statistics) and the two-point probability function $S_2$ (obtained using FFT) from the real morphology.
2. **Initial Generation:** Generate the initial 3D synthetic microstructure based on the target volume fraction $\phi_0$ and an initial Gaussian kernel $\sigma_0 = 1.0$.
3. **Synthetic $S_2^{\prime}$ Calculation:** Obtain the synthetic two-point probability function $S_2^{\prime}$ from the generated 3D synthetic microstructure using FFT.
4. **Optimization Problem:** Establish an optimization problem to minimize the mean absolute percentage error (MAPE) between the real ($S_2$) and synthetic ($S_2^{\prime}$) two-point probability functions.
5. **Final Generation:** Obtain the optimized Gaussian kernel ($\sigma$) and generate the final 3D synthetic RVE microstructure that matches the given real morphology.

Additionally, the python codes used in this work can be found on the [GitHub repository](https://github.com/zhanqi-syt/Microstructure-based-RVE-generation-from-2D-to-3D).

![图7](https://user-images.githubusercontent.com/116877222/229271220-e0396908-f1a7-4ea1-bdfa-ebec7f7803ec.png)
<br>**Fig. 1. Flowchart of the synthetic RVE generation from real microstructures.**<br>
<br>
<br>
![图8](https://user-images.githubusercontent.com/116877222/229278321-ba24951d-e08c-4070-ab8b-e08d1429010a.png)
<br>**Fig. 2. Microstructures and two-point probability functions of sintered metal particles: (a) real microstructure, (b) generated RVE, and (c) two-point probability functions of real microstructure and generated RVE.**<br>







