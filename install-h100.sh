pip3 install torch==2.8.0 torchvision==0.23.0
pip3 install transformers==4.57.1
pip3 install mosaicml-streaming
pip3 install datasets
pip3 install wandb
pip3 install pandas pyarrow
pip3 install multiprocess
FLASH_ATTENTION_SKIP_CUDA_BUILD="TRUE" pip3 install flash-attn==2.8.3 --no-build-isolation
pip3 install git+https://github.com/apple/ml-cross-entropy
pip3 install liger-kernel==0.6.2