# Final değerlendirme — test seti

- Final model: `E05_y11s_640_dfire` (val'e göre seçildi) · baseline: `E01_y11s_640`
- Güven eşikleri val'de seçildi (kare bazında yanlış alarm ≤ %1.0): fire 0.25, smoke 0.30
- Test: 4271 kare · FASDD 3148 · Heinola (görülmemiş yangın) 1123

## Standart ölçüler (mAP)

| test kümesi | sınıf | final mAP50 | final mAP50-95 | baseline mAP50 | baseline mAP50-95 |
|---|---|---|---|---|---|
| hepsi | all | 0.560 | 0.296 | 0.473 | 0.244 |
| hepsi | fire | 0.249 | 0.076 | 0.182 | 0.061 |
| hepsi | smoke | 0.871 | 0.517 | 0.764 | 0.428 |
| fasdd | all | 0.541 | 0.298 | 0.461 | 0.257 |
| fasdd | fire | 0.250 | 0.076 | 0.183 | 0.061 |
| fasdd | smoke | 0.831 | 0.521 | 0.740 | 0.452 |
| boreal | all | 0.939 | 0.516 | 0.816 | 0.398 |
| boreal | smoke | 0.939 | 0.516 | 0.816 | 0.398 |

## Kare bazında (val'de seçilen eşikle)

| sınıf | kaynak | final yakalama | final yanlış alarm | baseline yakalama | baseline yanlış alarm |
|---|---|---|---|---|---|
| fire | hepsi | %90.2 | %2.0 | %77.8 | %1.8 |
| fire | fasdd | %90.2 | %2.8 | %77.8 | %2.6 |
| fire | boreal | — | %0.0 | — | %0.1 |
| smoke | hepsi | %95.1 | %0.3 | %90.4 | %1.3 |
| smoke | fasdd | %95.8 | %0.4 | %84.3 | %0.1 |
| smoke | boreal | %94.0 | %0.0 | %99.1 | %12.0 |

## Sahne ortalaması ve piksel kapsama

| model              | sınıf   |   sahne sayısı |   sahne ort. yakalama % |   piksel R |   piksel P |
|:-------------------|:--------|---------------:|------------------------:|-----------:|-----------:|
| E05_y11s_640_dfire | fire    |             13 |                  79.689 |      0.738 |      0.871 |
| E05_y11s_640_dfire | smoke   |             51 |                  92.855 |      0.855 |      0.93  |
| E01_y11s_640       | fire    |             13 |                  64.798 |      0.822 |      0.731 |
| E01_y11s_640       | smoke   |             51 |                  85.543 |      0.87  |      0.824 |

Kaçırılan ateş karesi: 58 · yanlış alarm karesi: 79 · Heinola'da kaçırılan duman karesi: 54