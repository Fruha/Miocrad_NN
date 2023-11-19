## Description

Этот проект нацелен на разработку и обучение нейронной сети для выявления признаков инфаркта на основе данных ЭКГ из набора данных [PTB Diagnostic ECG](https://physionet.org/content/ptbdb/1.0.0/) Database.

## Installation
```bash
git clone https://github.com/Fruha/Miocrad_NN
cd Miocrad_NN
pip install -r requirements.txt
```

## Usage

### Training
```bash
python main.py -m experiment=experiment1,experiment2,experiment3,experiment4,experiment5,experiment6
pause
```

### Visualize
```bash
start "" "http://localhost:6006/"
tensorboard --logdir tb_logs --port 6006
pause
```