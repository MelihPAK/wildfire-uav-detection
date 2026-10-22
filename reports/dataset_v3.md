# Dataset v3 — firesmoke (v2 + D-Fire, sadece train)

- v2 + D-Fire 14569 görüntü (pozitif 11632, negatif 2937, NEG_FRAC 0.3)
- D-Fire toplam 21527 · val/test'e benzediği için atılan 105 (eşik τ−MARGIN = 0.916, τ = 0.936)
- Sınıf eşlemesi (isimden): {0: 'smoke', 1: 'fire'}
- val ve test v2 ile birebir aynı

|                     |   fire+smoke |   negatif |   sadece fire |   sadece smoke |   toplam |
|:--------------------|-------------:|----------:|--------------:|---------------:|---------:|
| ('train', 'boreal') |            0 |        39 |             0 |           3225 |     3264 |
| ('train', 'dfire')  |         4658 |      2937 |          1139 |           5835 |    14569 |
| ('train', 'fasdd')  |         6733 |      8430 |           140 |           3557 |    18860 |
| ('val', 'boreal')   |            0 |         0 |             0 |            562 |      562 |
| ('val', 'fasdd')    |          549 |      1758 |            20 |            762 |     3089 |
| ('test', 'boreal')  |            0 |       217 |             0 |            906 |     1123 |
| ('test', 'fasdd')   |          537 |      1798 |            52 |            761 |     3148 |