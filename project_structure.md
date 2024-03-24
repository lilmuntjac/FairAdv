# Project structure

```
FairAdv/
│
├── adversarial/
│   ├── losses/
│   │   ├── direct_loss.py
│   │   └── ... (other loss files)
│   ├── masks/
│   │   └── eyeglasses_mask_6percent.png
│   ├── perturbation_applier.py
│   ├── frame_applier.py
│   └── eyeglasses_applier.py
│
├── analysis/
│   └── conf_matrix_dist.py
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
│       └── celeba_pattern_tester.yml
│
├── dataloaders/
│   ├── dataloader.py
│   └── samplers.py
│
├── datasets/
│   ├── celeba_dataset.py
│   ├── celeba_xform_dataset.py
│   ├── fairface_dataset.py
│   ├── fairface_xform_dataset.py
│   └── ham10000_dataset.py
│
├── models/
│   ├── binary_model.py
│   └── multiclass_model.py
│
├── tests/
│   ├── test_appliers.py
│   └── test_pattern.py
│
├── utils/
│   ├── model_utils.py
│   └── utils.py
│
├── train.py
├── train_adversarial.py # deprecated
└── requirements.txt
```
