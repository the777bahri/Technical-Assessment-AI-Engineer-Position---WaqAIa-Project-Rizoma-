# Environment Setup Commands

## 1. Create the Conda environment

```bash
conda create -n Rizoma_Task -y python=3.11.9
```

## 2. Activate the environment

```bash
conda activate Rizoma_Task
```

## 3. Install Jupyter in your environment

```bash
conda install -y jupyter
```

or

```bash
pip install notebook
```

## 4. Link the kernel to Jupyter

```bash
python -m ipykernel install --user --name Rizoma_Task --display-name "Python (Rizoma_Task)"
```

## 5. Double-check everything works

List available kernels:

```bash
jupyter kernelspec list
```

Check that your environment (`Rizoma_Task`) appears in the list.

## 6. Install PyTorch

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
