# Dataset v2 — firesmoke (DINOv2 ile temizlenmiş)

- Görüntü: 30046 · grup: 211 · τ: 0.936 (otomatik, NEG_Q=0.99) · K: 64 · seed: 0
- Kalibrasyon: kesin farklı ≥ τ %1.1 · kesin aynı ≥ τ %99.9
- FASDD test ≥ τ: v1 1205 → v2 0 görüntü · val ≥ τ: 0
- v1 → v2 yer değiştiren görüntü: 7937

## Split × kaynak
|                     |   görüntü |   negatif |   fire_kutu |   smoke_kutu |
|:--------------------|----------:|----------:|------------:|-------------:|
| ('train', 'boreal') |      3264 |        39 |           0 |         3377 |
| ('train', 'fasdd')  |     18860 |      8430 |       29899 |        13671 |
| ('val', 'boreal')   |       562 |         0 |           0 |          568 |
| ('val', 'fasdd')    |      3089 |      1758 |        3955 |         1947 |
| ('test', 'boreal')  |      1123 |       217 |           0 |          917 |
| ('test', 'fasdd')   |      3148 |      1798 |        2453 |         1604 |

## Split × kaynak × kategori
|                     |   fire+smoke |   sadece fire |   sadece smoke |   negatif |   toplam |   negatif % |
|:--------------------|-------------:|--------------:|---------------:|----------:|---------:|------------:|
| ('boreal', 'test')  |            0 |             0 |            906 |       217 |     1123 |        19.3 |
| ('boreal', 'train') |            0 |             0 |           3225 |        39 |     3264 |         1.2 |
| ('boreal', 'val')   |            0 |             0 |            562 |         0 |      562 |         0   |
| ('fasdd', 'test')   |          537 |            52 |            761 |      1798 |     3148 |        57.1 |
| ('fasdd', 'train')  |         6733 |           140 |           3557 |      8430 |    18860 |        44.7 |
| ('fasdd', 'val')    |          549 |            20 |            762 |      1758 |     3089 |        56.9 |