# GA Logs
## Table of Contents
- [Pilot Experments](#pilot-experiments-on-problem-18)
- [F18 Fine-tuning](#f18-fine-tuning)
- [F19 Fine-tuning](#f19-fine-tuning)

## Pilot Experiments (on Problem 18)
Note: AUC in pilot experments are not standardized.\
seed = 42\
problem_dim = 30\
budget = 5000\
runs = 20

|Selection|pop_k|tour_k|tour_p|mut_r|update_r|AUC|avg. fitness|succ|min|max|median|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|prop|10|5|0.8|0.5|0.5|0.8105|3.6|4|3.06|3.91|3.66|
|rank|10|5|0.8|0.5|0.5|0.7800|3.57|1|2.9|5.42|3.57|
|tour|10|5|0.8|0.5|0.5|0.8094|3.63|2|3.06|4.21|3.55|
|mix|10|5|0.8|0.5|0.5|0.7984|3.61|3|3.06|4.21|3.55|
|mix|5|5|0.8|0.5|0.5|0.8195|3.6|3|3.24|4.21|3.44|
|mix|2|2|0.8|0.5|0.5|0.7919|3.52|2|3.06|3.91|3.55|
|mix|4|4|0.8|0.5|0.5|0.7896|3.53|2|3.06|3.91|3.55|
|mix, force child.f > origin|5|5|0.8|0.5|0.5|0.8092|3.72|1|3.24|6|3.55|

### Round 2
mix\
pop_k = 5\
tour_k = 5\
force child.f > origin

|mix_s|tour_p|mut_r|update_r|AUC|avg. fitness|succ|min|max|median|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|1-2-2|0.9|0.5|0.5|0.7973|3.63|3|3.06|4.21|3.63|
|1-1-3|0.8|0.5|0.5|0.8124|3.66|1|3.06|5.42|3.66|
|0.5-0.5-4|0.8|0.5|0.5|0.8093|3.77|1|2.9|5.42|3.66|

### Round 3
mix 1-1-3\
tour_p = 0.8

|mut_r|update_r|AUC|avg. fitness|succ|min|max|median|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|annealing, piecewise|0.5|0.8221|3.69|1|3.06|4.55|3.66|
|annealing, piecewise, 0.1, 1/$l$|0.4|0.8418|3.84|2|3.24|4.55|3.66

## F18 Fine-tuning
dim=50\
tounament_k = pop_k\
tournament_p = 0.8\
mix 2-1-2, prop->rank->tour

**Notes** 
1. AUC_u represents calculating the AUC with default settings.
2. ECDF settings: **not** use "scale x axis log10".
3. ERT settings: **not** use "scale x axis log10". Use "scale y axis log10".
4. ETV settings: use "scale x axis log10". Use "scale y axis log10".
5. Save figures: pdf and png. ECDF, ERT, ETV.

### Population Size
|p_size|AUC|avg. fitness|max|median|
|:----:|:---:|:---:|:---:|:---:|
|2  |0.379332727272727  |2.62  |3.12   |2.62   |
|5  |0.419120454545454  |3.23  |3.71   |3.28   |
|10 |0.451300909090909  |3.70  |5.36   |3.71   |
|20 |0.463674090909091  |3.90  |4.72   |3.89   |
|25 |0.445823636363636  |3.72  |4.45   |3.71   |

psize = 20

### Mutation Rate
|m_rate|AUC|avg. fitness|max|median|
|:---:|:---:|:---:|:---:|:---:|
|0.1        |0.39148            |3.01  |3.71   |2.94   |
|0.02       |0.495772272727273  |3.81  |5.36   |3.71   |
|0.05       |0.416986818181818  |3.36  |4.21   |3.35   |
|piecewise  |0.463674090909091  |3.90  |4.72   |3.89   |

mrate = piecewise

### Update Ratio
|u_rate|AUC|avg. fitness|max|median|
|:---:|:---:|:---:|:---:|:---:|
|0.2    |0.418279090909091  |3.53  |4.58   |3.43   |
|0.4    |0.463674090909091  |3.90  |4.72   |3.89   |
|0.8    |0.442728636363636  |3.35  |4.21   |3.28   |
|1.0    |0.413140909090909  |2.99  |3.71   |2.94   |

urate = 0.4

### Final Results
|AUC|avg. fitness|max|median|
|:---:|:---:|:---:|:---:|
|0.463674090909091  |3.90  |4.72   |3.89   |

## F19 Fine-tuning
### Population Size
|p_size|AUC|avg. fitness|max|median|
|:----:|:---:|:---:|:---:|:---:|
|2  |0.366114545454545  |31.1  |38 |30 |
|5  |0.623161818181818  |39.8  |46 |41 |
|10 |0.726854090909091  |44.1  |48 |44 |
|20 |0.762886363636364  |46.5  |48 |46 |
|25 |0.750179090909091  |46.0  |50 |46 |

psize = 20

### Mutation Rate
|m_rate|AUC|avg. fitness|max|median|
|:---:|:---:|:---:|:---:|:---:|
|0.1        |0.614391818181818  |41.9  |44 |42 |
|0.02       |0.762886363636364  |46.5  |48 |46 |
|0.05       |0.691100454545455  |45.3  |48 |46 |
|piecewise  |0.668567727272727  |45.0  |48 |46 |

mrate = 0.02

### Update Ratio
|u_rate|AUC|avg. fitness|max|median|
|:---:|:---:|:---:|:---:|:---:|
|0.2    |0.68591            |44.9   |48 |45 |
|0.4    |0.762886363636364  |46.5   |48 |46 |
|0.8    |0.698147272727273  |42.6   |46 |42 |
|1.0    |0.688687272727273  |42.3   |46 |42 |

urate = 0.4

### Final Results
|AUC|avg. fitness|max|median|
|:---:|:---:|:---:|:---:|
|0.762886363636364  |46.5   |48 |46 |
