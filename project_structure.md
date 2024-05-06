# Project structure

```
FairAdv/
│
├── adversarial/
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── binary_base.py
│   │   ├── binary_combinedloss.py
│   │   ├── constraint.py
│   │   ├── equimask.py
│   │   ├── multiclass_base.py
│   │   ├── perturbed_optimizer.py
│   │   └── ... (other loss files)
│   ├── masks/
│   │   └── eyeglasses_mask_6percent.png
│   ├── perturbation_applier.py
│   ├── frame_applier.py
│   └── eyeglasses_applier.py
│
├── analysis/
│   ├── conf_matrix_dist.py # deprecated
│   ├── epoch_analyzer.py
│   └── plot_stats.py
│
├── config/ 
│   ├── fairness_attack
│   │   └── celeba_pert_at_male.yml
│   ├── mfd_training
│   │   └── celeba_at_male.yml
│   ├── model_training
│   │   ├── celeba_39_male.yml
│   │   ├── celeba_at_male.yml
│   │   ├── celeba_bn_male.yml
│   │   ├── celeba_bu_male.yml
│   │   ├── celeba_hc_male.yml
│   │   ├── celeba_ov_male.yml
│   │   ├── celeba_yg_male.yml
│   │   ├── celeba.yml
│   │   ├── fairface.yml
│   │   └── ham10000.yml
│   └── testing
│       ├── celeba_39m_cm.yml
│       ├── celeba_applier.yml
│       ├── celeba_pattern_tester.yml
│       └── celeba_stats.yml
│
├── data
│   ├── datasets
│   │   ├── celeba_dataset.py
│   │   ├── celeba_xform_dataset.py
│   │   ├── fairface_dataset.py
│   │   ├── fairface_xform_dataset.py
│   │   └── ham10000_dataset.py
│   └── loaders
│       ├── dataloader.py
│       └── samplers.py
│
├── models/
│   └── generic_model.py
│
├── tests/
│   ├── test_checkpoint.py
│   └── test_pattern_stats.py
│
├── training/
│   ├── train_fscl_supcon.py
│   ├── train_generic.py
│   ├── train_mfd.py
│   └── train_pattern.py
│
├── utils/
│   ├── config_utils.py
│   ├── metrics_utils.py
│   └── training_utils.py
│
├── requirements.txt
└── train.py
```
