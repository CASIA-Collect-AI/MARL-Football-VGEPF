# Vision-Based Generic Potential Function (V-GEPF)

[English](README.md) | [中文](README_CN.md)

[![AAAI 2025](https://img.shields.io/badge/AAAI-2025-blue)](https://aaai.org/conference/aaai/aaai-25/)
[![arXiv](https://img.shields.io/badge/arXiv-2502.13430-red)](https://arxiv.org/abs/2502.13430)
[![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## 🏛️ About This Repository

This repository is maintained by **[CASIA-Collect-AI](https://github.com/CASIA-Collect-AI)** as part of a curated collection of high-quality MARL research code.

📌 **Original Repository (Recommended):** [Marroh/V-GEPF-official-code](https://github.com/Marroh/V-GEPF-official-code)
⭐ **If this work is helpful, please Star the original repository to support the authors!**

> **Team:** Intelligent Flight Technology Team (Swarm Intelligence Group), Institute of Automation, Chinese Academy of Sciences (CASIA), led by Prof. Zhiqiang Pu.
> CASIA-Collect-AI curates and maintains high-quality open-source research code in MARL, LLM, and robotics.

---

Official implementation of **Vision-Based Generic Potential Function for Policy Alignment in Multi-Agent Reinforcement Learning** (AAAI 2025).

**Authors:** Hao Ma, Shijie Wang, Zhiqiang Pu, Siyao Zhao, Xiaolin Ai
**Affiliations:** Institute of Automation, Chinese Academy of Sciences; University of Chinese Academy of Sciences

---

## Abstract

Multi-agent reinforcement learning (MARL) in complex environments such as Google Research Football (GRF) faces a core challenge: agents often converge to locally optimal but tactically poor policies — lacking coordination, positioning, and human-intuitive play styles.

This paper proposes **V-GEPF**, a framework that leverages Vision-Language Models (VLMs) to generate **potential-based reward shaping** for MARL policy alignment. The key insight: by encoding game states as images and measuring cosine similarity with human-language instructions, V-GEPF provides a generic, human-interpretable reward signal that steers agent behavior toward desired strategies without environment-specific reward engineering.

**Three core contributions:**
1. A VLM-based **generic potential function** using cosine distance between image and text encodings
2. An adaptive **vLLM-based selector** that automatically chooses appropriate potential functions based on video replays of past episodes
3. Demonstrated policy alignment in GRF 11-vs-11, outperforming PPO/MAPPO/HAPPO baselines

---

## 📖 Paper Deep Dive

### The Problem: Policy Misalignment in Complex MARL

Standard MARL training (MAPPO, HAPPO) in the GRF 11-vs-11 scenario produces policies that win games through simple tactics — but exhibit poor tactical quality:

<div align="center">
  <img src="imgs/fig_motivation_a.png" width="340" style="display:inline-block; margin:4px;">
  <img src="imgs/fig_motivation_b.png" width="340" style="display:inline-block; margin:4px;">
</div>

*(Left) MAPPO: non-ball-holding players fail to occupy valuable space. (Right) HAPPO: players lack coordinated positioning to create passing opportunities.*

The reward signal alone cannot encode strategic quality — a goal scored through chaotic play counts the same as a team goal from coordinated build-up. V-GEPF addresses this with visually grounded reward shaping.

---

### V-GEPF Framework

<div align="center"><img src="imgs/fig_framework.png" width="750"></div>

*V-GEPF framework: game state image + human instruction → CLIP cosine distance → potential reward → policy alignment. A vLLM watches episode replays and adaptively selects which potential function to apply.*

**Two-stage design:**

**Stage 1 — CLIP-based Potential Function**
- Game state is rendered as an image $s_t^G$
- Human instruction $l$ (e.g., "maintain triangle passing shape") is encoded as text
- Cosine distance between image and text embeddings defines the potential: $\phi(s_t | l) = \text{cos}(\text{enc}_\text{img}(s_t^G), \text{enc}_\text{txt}(l))$
- Potential-based reward shaping: $r'_t = r_t + \gamma \phi(s_{t+1}) - \phi(s_t)$

**Stage 2 — vLLM Adaptive Selector**
- At the start of each training phase, a vLLM (e.g., MiniCPM-o) watches the video replay of the last episode
- Given a pool of potential functions and reflection on previous selections, the vLLM chooses the most appropriate instruction
- This enables curriculum-style learning: early phases focus on basic positioning; later phases target advanced tactics

---

### Adaptive Potential Function Selection

<div align="center"><img src="imgs/fig_potential.png" width="620"></div>

*Potential function curves during training. Six VLM-based potential functions are selected sequentially by the vLLM, each guiding the agents toward increasingly sophisticated tactical behaviors.*

The vLLM selects from a pool of human-defined instructions (attack, defend, formation, dribble, etc.) based on video evidence of current policy weaknesses — mimicking how a human coach adjusts training focus.

---

### Experimental Results

**Environment:** Google Research Football (GRF)
- **11-vs-11 Easy**: Standard opponent AI
- **11-vs-11 Hard**: Stronger opponent AI
- **Academy Scenarios**: counterattack_easy, counterattack_hard (transfer experiments)

<div align="center"><img src="imgs/fig_results_easy.png" width="580"></div>

*Win rate curves in GRF 11-vs-11 Easy. V-GEPF + MAPPO consistently outperforms MAPPO, HAPPO, and baseline variants.*

<div align="center"><img src="imgs/fig_results_hard.png" width="580"></div>

*Win rate curves in GRF 11-vs-11 Hard. V-GEPF maintains its advantage under stronger opponents.*

**Key findings:**
- V-GEPF improves win rates across all GRF scenarios
- Policies exhibit visually verifiable tactical improvement (coordinated passes, structured formations)
- The xT-based fallback (non-VLM) also improves over baseline, validating the potential-shaping approach

---

### Visual Policy Analysis

<div align="center"><img src="imgs/fig_visual_analysis.png" width="750"></div>

*A visual analysis of V-GEPF policy in an offensive phase. Agents form coordinated positioning and coherent passing combinations aligned with human football intuition — a direct result of VLM-guided potential shaping.*

---

## Installation

```bash
conda create -n v-gepf python=3.9
conda activate v-gepf
```

1. **Install Google Research Football** following the [official repo](https://github.com/google-research/football)

2. **Install base framework:**
   ```bash
   pip install -e .
   ```

3. **Install MiniCPM-o** following the [official guide](https://github.com/OpenBMB/MiniCPM-o)

---

## Quick Start

### Configuration

Set model paths in `light_malib/vlm/vlm_critic.py`:
```python
class LocalMiniCPM:
    def __init__(self, model_path='/path/to/MiniCPM-Llama3-V-2_5', device='auto'):
        ...

class CLIPCritic:
    def __init__(self):
        self.clip, self.preprocess = clip.load("/path/to/CLIP/RN50.pt", device="cuda")
```

(Optional) Set OpenAI API key in `light_malib/vlm/utils.py` for GPT-based VLMs.

### Run Experiments

```bash
# V-GEPF with MAPPO in 11-vs-11
sh scripts/run_vllm_mappo_11v11.sh

# xT-based potential (non-VLM alternative)
# Uncomment xT block in light_malib/envs/gr_football/env.py step()
```

---

## Core Implementation

| File | Description |
|------|-------------|
| `light_malib/vlm/vlm_critic.py` | VLM critics: `MiniCPMCritic`, `CLIPCritic` — compute potential rewards from game state images |
| `light_malib/vlm/utils.py` | `RealTimeDrawer` (state → image), `vLLMAgent` + `vLLMMemory` (adaptive selector) |
| `light_malib/algorithm/mappo/trainer.py` | MAPPO training loop with potential reward integration and Ray parallelization |

---

## Citation

```bibtex
@inproceedings{ma2025vision,
  title={Vision-Based Generic Potential Function for Policy Alignment in Multi-Agent Reinforcement Learning},
  author={Ma, Hao and Wang, Shijie and Pu, Zhiqiang and Zhao, Siyao and Ai, Xiaolin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={18},
  pages={19287--19295},
  year={2025}
}
```

---

## Contact

- **First Author:** Hao Ma — CASIA
- **Corresponding Author:** zhiqiang.pu@ia.ac.cn (Prof. Zhiqiang Pu)
