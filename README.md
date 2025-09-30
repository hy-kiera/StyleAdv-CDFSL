# StyleAdv-CDFSL with RL-based Hyperparameter Tuning

This repository implements a **reinforcement learning–based approach** for StyleAdv: Meta Style Adversarial Training for Cross-Domain Few-Shot Learning [[Paper](https://arxiv.org/pdf/2302.09309)]

Traditional adversarial style generation methods often rely on manually tuned hyperparameters for the perturbation strength. To address this limitation, this work formulates the selection of $\epsilon$ as a **contextual bandit problem**. By leveraging the **REINFORCE algorithm**, the method dynamically learns the optimal $\epsilon$ values during training.

# Results of 5-way 1-shot/5-shot tasks
“FT” indicates whether the finetuning stage is employed.

| n-Shot   | FT | ChestX            | ISIC              | EuroSAT           | CropD             | Avg   |
|----------|----|-----------------|-----------------|-----------------|-----------------|-------|
| 1-Shot   | N  | 22.73 ± 0.36    | 33.38 ± 0.52    | 71.54 ± 0.85    | 76.75 ± 0.81    | 51.1  |
|          | Y  | 22.55 ± 0.34    | 34.42 ± 0.55    | 72.28 ± 0.81    | 79.50 ± 0.68    | 52.19 |
| 5-Shot   | N  | 24.95 ± 0.36    | 44.67 ± 0.50    | 88.33 ± 0.51    | 93.52 ± 0.39    | 62.87 |
|          | Y  | 26.07 ± 0.36    | 52.28 ± 0.55    | 89.49 ± 0.45    | 96.22 ± 0.29    | 66.02 |
