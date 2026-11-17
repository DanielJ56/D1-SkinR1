# Skin-R1

### D1 Reproduction 

Original Repo: https://github.com/l593191569/Skin-R1


### Setup

```bash
!pip install -q torch==2.8.0 torchvision==0.23.0 transformers==4.52.3 peft==0.17.0 accelerate==1.4.0
!git clone -q https://github.com/l593191569/Skin-R1.git /content/Skin-R1
```

### Run

Change line 10 in skinr1_feeding.py IMAGE_PATH to local path of images

```bash
python skinr1_feeding.py
```

Outputs results to Master_Results.csv. Upon resuming, skips rows in Masters_Results.csv that are already filled.