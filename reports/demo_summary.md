# Demo ve export özeti

- ONNX: `firesmoke_y11s_640.onnx` (37.9 MB), girdi 640×640
- Doğrulama (ONNX + predict.py ↔ PyTorch): görüntü 200 · kutu (ONNX) 491 · kutu (PyTorch) 485 · eşleşen % 84.3 · ort. güven farkı 0.0208 · ONNX CPU ms/görüntü 31.7 · PyTorch GPU ms/görüntü 29.7
- Zaman filtresi: son 5 karenin en az 3'sında tespit → alarm · eşikler fire 0.25, smoke 0.3

## Heinola (görülmemiş yangın), olay bazında

| ölçüm                                             |   ham kare |   filtreli alarm |
|:--------------------------------------------------|-----------:|-----------------:|
| duman yakalama %                                  |       94.5 |             94.5 |
| duman yanlış alarm %                              |        0   |              3.7 |
| ateş yanlış alarm % (Heinola'da ateş etiketi yok) |        0   |              0   |

| video                   |   kare |   ilk dumanlı kare |   ilk alarm (kare sonra) | alarm hiç yok   |
|:------------------------|-------:|-------------------:|-------------------------:|:----------------|
| boreal_heinola_DJI_0025 |    201 |                  0 |                        2 | False           |
| boreal_heinola_DJI_0028 |    267 |                  9 |                        4 | False           |
| boreal_heinola_DJI_0029 |    106 |                  0 |                        2 | False           |
| boreal_heinola_DJI_0030 |    143 |                  0 |                        2 | False           |
| boreal_heinola_DJI_0033 |    230 |                  0 |                        2 | False           |
| boreal_heinola_DJI_0036 |     45 |                  0 |                        2 | False           |
| boreal_heinola_DJI_0038 |     10 |                  0 |                        2 | False           |
| boreal_heinola_DJI_0048 |     62 |                 41 |                      nan | True            |

- Demo: `demo_heinola.mp4` (841 kare, 4 video) · `assets/demo_heinola.gif`