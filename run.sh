#!/bin/bash

python train.py config/fscl_training/supcon/cb_at_m_b064_t0d005.yml
python train.py config/fscl_training/supcon/cb_at_m_b064_t0d010.yml
python train.py config/fscl_training/supcon/cb_at_m_b064_t0d020.yml
python train.py config/fscl_training/supcon/cb_at_m_b064_t0d030.yml
python train.py config/fscl_training/supcon/cb_at_m_b064_t0d040.yml

python train.py config/fscl_training/supcon/cb_at_m_b128_t0d005.yml
python train.py config/fscl_training/supcon/cb_at_m_b128_t0d010.yml
python train.py config/fscl_training/supcon/cb_at_m_b128_t0d020.yml
python train.py config/fscl_training/supcon/cb_at_m_b128_t0d030.yml
python train.py config/fscl_training/supcon/cb_at_m_b128_t0d040.yml

python train.py config/fscl_training/supcon/cb_at_m_b256_t0d005.yml
python train.py config/fscl_training/supcon/cb_at_m_b256_t0d010.yml
python train.py config/fscl_training/supcon/cb_at_m_b256_t0d020.yml
python train.py config/fscl_training/supcon/cb_at_m_b256_t0d030.yml
python train.py config/fscl_training/supcon/cb_at_m_b256_t0d040.yml