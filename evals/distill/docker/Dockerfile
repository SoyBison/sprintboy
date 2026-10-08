# Training image for DJ Laya: PyTorch with CUDA 12.8 (the 3090's driver is 590)
# plus laya, whose torch>=2.0 pin keeps this torch rather than pulling a CPU one.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
RUN pip install --no-cache-dir --root-user-action=ignore "laya[serve]==0.4.0" onnxruntime onnx
# laya 0.4.0 scores the base checkpoint on the eval slice before train() moves
# it to the GPU, which fails on CUDA ("found at least two devices"). Move the
# model to the device as soon as it is loaded.
RUN sed -i 's/^    model, tok, cfg = load_checkpoint(model_dir)$/    model, tok, cfg = load_checkpoint(model_dir)\n    model.to(dev)/' \
        /opt/conda/lib/python3.11/site-packages/laya/train.py \
 && grep -q '^    model.to(dev)$' /opt/conda/lib/python3.11/site-packages/laya/train.py
ENV HF_HOME=/work/hf
WORKDIR /work
