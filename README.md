# Estimation of Distribution Algorithm to optimize the parameters of a QAOA

This repository contains the experiments and code used for the work that was presented in The Genetic and Evolutionary Computation Congress 2022 (GECCo'22) in Boston.

## Abstract
Variational quantum algorithms (VQAs) offer some promising characteristics for carrying out optimization tasks in noisy intermediate-scale quantum devices. These algorithms aim to minimize a cost function by optimizing the parameters of a quantum parametric circuit. Thus, the overall performance of these algorithms, heavily depends on the classical optimizer which sets the parameters. In the last years, some gradient-based and gradient-free approaches have been applied to optimize the parameters of the quantum circuit. In this work, we follow the second approach and propose the use of estimation of distribution algorithms for the parameter optimization in a specific case of VQAs, the quantum approximate optimization algorithm. Our results show an statistically significant improvement of the cost function minimization compared to traditional optimizers.

## Citation
Soloviev, V. P., Larrañaga, P., & Bielza, C. (2022, July). Quantum parametric circuit optimization with estimation of distribution algorithms. In Proceedings of the Genetic and Evolutionary Computation Conference Companion (pp. 2247-2250).

@inproceedings{soloviev2022quantum,
  title={Quantum parametric circuit optimization with estimation of distribution algorithms},
  author={Soloviev, Vicente P and Larra{\~n}aga, Pedro and Bielza, Concha},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference Companion},
  pages={2247--2250},
  year={2022}
}
