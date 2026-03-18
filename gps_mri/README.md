# GPS for Semi-Supervised MRI Reconstruction

이 구현은 요청한 최종 구조를 기준으로 작성했습니다.

## Directory

```text
gps_mri/
  data/
    fastmri_dataset.py
    transforms.py
  models/
    varnet.py
    sensitivity_model.py
    gps_recon_model.py
  training/
    train_gps_recon.py
    losses.py
  scripts/
    train_fastmri.py
    evaluate.py
```

## What is implemented

- **E2E-VarNet 기반 단일 네트워크**
  - sensitivity U-Net + 8 cascades + soft DC (learnable lambda)
  - multi-coil forward / adjoint 연산 (`A = MFC`) 포함
- **GPS 구조**
  - teacher + student1 + student2 (독립 파라미터)
  - teacher pretrain / student GPS+CPS / teacher feedback 단계별 루프
- **Losses**
  - `L_recon = L1 + 0.5*(1-SSIM)`
  - `L_dc` (forward consistency)
  - `L_gps`, `L_cps`, `L_feedback`
  - exponential ramp (`10k`, feedback `5k`) 지원
- **데이터**
  - fastMRI HDF5 volume-level labeled/unlabeled split
  - retrospective undersampling mask 생성 (ACS 8%)

## Notes

- 이 환경에는 `torch`가 없어 실행 검증은 py_compile 수준으로 제한됩니다.
- `fastmri_dataset.py`는 일반적인 fastMRI HDF5 스키마를 가정하므로,
  실제 파일 스키마 차이가 있으면 키 이름(`kspace`, `reconstruction_rss`)을 조정해야 합니다.
