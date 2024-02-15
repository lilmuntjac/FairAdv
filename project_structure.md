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
│   └── generate_diagrams.py
│
├── config/ 
│   ├── fairness_attack
│   │   └── celeba_pert.yml
│   ├── model_training
│   │   ├── celeba.yml
│   │   └── fairface.yml
│   └── testing
│       ├── celeba_applier.yml
│       └── celeba_pattern_tester.yml
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
│   ├── test_pattern.py
│   └── explore_celeba.py
│
├── utils/
│   ├── utils.py
│   └── data_loader.py
│
├── train.py
├── train_adversarial.py
└── requirements.txt
```
