# Quantizing Flux 2 Klein 4B with SVDQuant

Flux 2 Klein 4B shares the same `FluxTransformer2DModel` architecture as Flux 1 (schnell/dev) but with 4B parameters. The quantization workflow follows the same 3-step process as other diffusion models.

All commands should be run from the `examples/diffusion/` directory.

## Step 1: Generate Reference Baselines

Generate unquantized BF16 outputs for similarity metrics evaluation:

```bash
python -m deepcompressor.app.diffusion.ptq configs/model/flux.2-klein.yaml --output-dirname reference
```

## Step 2: Collect Calibration Dataset

Randomly sample 128 prompts from COCO Captions 2024 for calibration:

```bash
python -m deepcompressor.app.diffusion.dataset.collect.calib \
    configs/model/flux.2-klein.yaml configs/collect/qdiff.yaml
```

## Step 3: Run INT4 SVDQuant Quantization + Evaluation

```bash
python -m deepcompressor.app.diffusion.ptq \
    configs/model/flux.2-klein.yaml configs/svdquant/int4.yaml \
    --eval-benchmarks MJHQ --eval-num-samples 1024
```

### Optional Variants

**Faster quantization** (fewer calib samples, coarser grid search):

```bash
python -m deepcompressor.app.diffusion.ptq \
    configs/model/flux.2-klein.yaml configs/svdquant/int4.yaml configs/svdquant/fast.yaml \
    --eval-benchmarks MJHQ --eval-num-samples 1024
```

**GPTQ refinement** after SVDQuant:

```bash
python -m deepcompressor.app.diffusion.ptq \
    configs/model/flux.2-klein.yaml configs/svdquant/int4.yaml configs/svdquant/gptq.yaml \
    --eval-benchmarks MJHQ --eval-num-samples 1024
```

**Save quantized checkpoint** for Nunchaku deployment:

```bash
python -m deepcompressor.app.diffusion.ptq \
    configs/model/flux.2-klein.yaml configs/svdquant/int4.yaml \
    --eval-benchmarks MJHQ --eval-num-samples 1024 --save-model true
```

## Deployment with Nunchaku

If a checkpoint was saved, convert it for the Nunchaku inference engine:

```bash
python -m deepcompressor.backend.nunchaku.convert \
    --quant-path /PATH/TO/CHECKPOINT/DIR \
    --output-root /PATH/TO/OUTPUT/ROOT \
    --model-name flux.2-klein
```

Then switch to the Nunchaku conda environment and follow the [Nunchaku](https://github.com/mit-han-lab/nunchaku) documentation for GPU deployment.
