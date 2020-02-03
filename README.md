# FastMRI - Neural Computation Group Project

Long acquisition times in fully-sampled MRI lead to low patient throughput, problems with patient comfort and compliance, artefacts from patient motion, and high examination costs. Reducing acquisition time using under-sampling acceleration rates (4 fold and 8 fold masks), helps to mitigate these issues, however, this comes at the const of reconstructed image quality.

The project aims to improve the viability of the more efficient under-sampling strategy by designing a system that maximises the quality of under-sampled reconstructions. High-quality reconstructions are images that closely align with the ground truth, the fully-sampled image. SSIM (Structural Similarity Index Measure) is used to evaluate the success of the model's output against the ground truth image.

The model implemented was a U-Net, with RMSprop as an optimizer. The majority of this project consisted of experimenting with various hyperparameters and features, to gain a better understanding of the fundamental components and concepts of a neural network.
