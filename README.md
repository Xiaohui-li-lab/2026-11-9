# Chiral Coarse-Quantum Framework — Computational Toolkit

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the reference implementation for the computational
methods proposed in:

> **Human Consciousness Can Construct the Universe: A Chiral Coarse-Quantum
> Framework and Testable Predictions**  
> Xiaohui Li (Independent Researcher, Baoji, China)  
> ORCID: 0009-0003-8233-7191

Three main tools are provided:

1. **Chiral spin-network simulation** (`src/spin_network.py` and `cpp/spin_network.cpp`)
2. **CMB chirality estimator** (`src/cmb_chirality.py`)
3. **Linguistic chirality index** (`src/linguistic_chirality.py`)

## Important scientific caveat

The underlying theory is exploratory and highly speculative. The code
implements *testable estimators* derived from the framework; it does not prove
the theory. Results should be interpreted critically.

## Quick start

```bash
# Clone the repository
git clone https://github.com/xiaohui-li-lab/chiral-coarse-quantum.git
cd chiral-coarse-quantum

# Install Python dependencies
pip install -r requirements.txt



使用说明
将上述所有文件按目录结构放入仓库。
安装依赖：
Bash
pip install -r requirements.txt
运行示例：
Bash
python examples/run_spin_demo.py
python examples/run_cmb_demo.py
python examples/run_linguistic_demo.py
运行测试：
Bash
pytest tests/
编译并运行 C++ 版本：
Bash
cd cpp
make
./spin_network --N 100 --sweeps 100000

# Run demos
python examples/run_spin_demo.py
python examples/run_cmb_demo.py
python examples/run_linguistic_demo.py
