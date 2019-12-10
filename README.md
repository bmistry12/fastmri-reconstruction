# FastMRI - Neural Computation Group Project

module load neural-comp
pip install --user torchsummary

module load neural-comp
jupyter-notebook

# Hyperparameters
epochs = 30
dropout_prob = 0.01
learning_rate = 0.001
weight_decay = 0.0
step_size = 15
lr_gamma = 0.1 # change in learning rate
num_pool_layers = 4

SSIM = 0.43435262911242406
