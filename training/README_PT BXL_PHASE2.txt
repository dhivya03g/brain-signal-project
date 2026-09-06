# PTB-XL Phase 2 pipeline

This package contains only the new PTB-XL pipeline. It does not modify the existing MIT-BIH CNN.

Run from repository root:

```powershell
python training/train_ptbxl_cnn.py --validate-only --limit 12
python test_ptbxl_cnn.py
```

Full training, only after the limited validation is reviewed:

```powershell
python training/train_ptbxl_cnn.py --train
```

The dataset is expected at `datasets/ptb-xl/` and is never copied by these scripts.
