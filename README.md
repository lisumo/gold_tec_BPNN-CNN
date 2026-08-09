# gold2tec_BPNN+CNN_Fusion

#### 介绍
双分支BPNN+CNN模型电离层TEC建模 v1.0

Yiming Li, Yibin Yao, Liang Zhang,

Preliminary verification of ionospheric TEC modeling based on GOLD FUV observations and a dual-branch neural network,

Advances in Space Research,Volume 78, Issue 5,2026,Pages 4825-4839,ISSN 0273-1177,
https://doi.org/10.1016/j.asr.2026.06.030.(https://www.sciencedirect.com/science/article/pii/S0273117726008033)

Abstract: Ionospheric Total Electron Content (TEC) modeling is crucial for correcting ionospheric delays and characterizing space weather variations. However, existing ionospheric TEC models rely heavily on ground-based GNSS networks, leading to observation gaps and accuracy degradation over vast oceanic regions where stations are sparse. The Far Ultraviolet (FUV) imaging data from the NASA GOLD mission provide a new data source for ionospheric TEC modeling. In this study, we propose a dual-branch fusion model integrating a Back Propagation Neural Network (BPNN) and a Convolutional Neural Network (CNN), which achieves the reconstruction of ionospheric TEC in the observation region by introducing the physical constraints of the IGRF geomagnetic model. Validation experiments based on the 2023 dataset indicate that the dual-branch fusion model outperforms the single-branch BPNN and CNN models, yielding a Root Mean Square Error (RMSE) of 7.97 TECU on the full test set, which represents improvements of 6.01% and 12.67% over the single-branch CNN and BPNN models, respectively. Independent validation using Jason-3 satellite altimetry data further confirmed the superiority of the fusion model in unsupervised ocean regions. The advantage of the fusion model lies in fully utilizing the merits of both branches: the BPNN branch leverages direct pixel-wise inversion of ultraviolet imaging to preserve the high-frequency details of the ionosphere, while the CNN branch performs macroscopic feature extraction, effectively suppressing high-frequency observation noise and maintaining the continuity of macro-structures such as the Equatorial Ionization Anomaly (EIA) double crests. This study validates the feasibility of retrieving ionospheric TEC using GOLD FUV imaging.

Keywords: TEC modeling; GOLD mission; Neural network; Data fusion
