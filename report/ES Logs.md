# ES Logs


## F18 Fine-tuning
**Notes** 
1. AUC_u represents calculating the AUC with default settings.
2. ECDF settings: **not** use "scale x axis log10".
3. ERT settings: **not** use "scale x axis log10". Use "scale y axis log10".
4. ETV settings: use "scale x axis log10". Use "scale y axis log10".
5. Save figures: pdf and png. ECDF, ERT, ETV.

### POPULATION SIZE
|PS|AUC|avg. fitness|max|median|
|:----:|:---:|:---:|:---:|:---:|
|2  | 0.416192 |     2.81     | 3.25 |  2.78  |
|  5   | 0.427722 |     2.81     | 3.46 |  2.76  |
|  10  | 0.418592 |     2.77     | 3.99 |  2.74  |
|  20  | 0.40538  |     2.75     | 3.71 |  2.74  |
|  50  | 0.424969 |     2.81     | 3.8  |  2.69  |

![image-20231128004727728](C:\Users\Hexiang\AppData\Roaming\Typora\typora-user-images\image-20231128004727728.png)

![image-20231128004738696](C:\Users\Hexiang\AppData\Roaming\Typora\typora-user-images\image-20231128004738696.png)



### STEP SIZE
|SS|AUC|avg. fitness|max|median|
|:---:|:---:|:---:|:---:|:---:|
| 0.1  | 0.442211  |     3.02     | 3.71 |  2.94  |
| 0.3  | 0.4177255 |     2.78     | 3.18 |  2.74  |
| 0.5  |  0.40538  |     2.75     | 3.71 |  2.74  |
| 0.7  | 0.4092055 |     2.76     | 3.25 |  2.71  |
| 0.9  | 0.4271635 |     2.78     | 3.25 |  2.78  |

![image-20231128005235078](C:\Users\Hexiang\AppData\Roaming\Typora\typora-user-images\image-20231128005235078.png)

![image-20231128005226532](C:\Users\Hexiang\AppData\Roaming\Typora\typora-user-images\image-20231128005226532.png)



### OFFSPRING SIZE
|OS|AUC|avg. fitness|max|median|
|:---:|:---:|:---:|:---:|:---:|
|  20  | 0.4061185 |     2.75     | 3.71 |  2.74  |
|  40  | 0.417048  |     2.8      | 3.71 |  2.71  |
|  60  | 0.4330125 |     2.85     | 3.54 |  2.81  |
|  80  | 0.4202155 |     2.76     | 3.25 |  2.74  |
| 100  | 0.4289375 |   2.812.81   | 3.32 |  2.78  |

![image-20231128005909114](C:\Users\Hexiang\AppData\Roaming\Typora\typora-user-images\image-20231128005909114.png)

![image-20231128010055454](C:\Users\Hexiang\AppData\Roaming\Typora\typora-user-images\image-20231128010055454.png)



### Final Results

|AUC|avg. fitness|max|median|
|:---:|:---:|:---:|:---:|
|      |              |      |        |

## F19 Fine-tuning
### POPULATION SIZE
|PS|AUC|avg. fitness|max|median|
|:----:|:---:|:---:|:---:|:---:|
|  2   | 0.693592 |      43      |  46  |   42   |
|  5   | 0.680788 |     42.3     |  46  |   42   |
|  10  | 0.687674 |     42.8     |  46  |   42   |
|  20  | 0.700275 |      43      |  46  |   43   |
|  50  | 0.672831 |     42.9     |  46  |   42   |

![image-20231128003523928](C:\Users\Hexiang\AppData\Roaming\Typora\typora-user-images\image-20231128003523928.png)

![image-20231128003640296](C:\Users\Hexiang\AppData\Roaming\Typora\typora-user-images\image-20231128003640296.png)



### STEP SIZE
|SS|AUC|avg. fitness|max|median|
|:---:|:---:|:---:|:---:|:---:|
| 0.1  | 0.653649 |     40.8     |  44  |   42   |
| 0.3  | 0.68339  |     42.4     |  44  |   42   |
| 0.5  | 0.700275 |      43      |  46  |   43   |
| 0.7  | 0.694355 |     43.3     |  46  |   44   |
| 0.9  | 0.690528 |      43      |  46  |   44   |

![image-20231128003438294](C:\Users\Hexiang\AppData\Roaming\Typora\typora-user-images\image-20231128003438294.png)

![image-20231128002914476](C:\Users\Hexiang\AppData\Roaming\Typora\typora-user-images\image-20231128002914476.png)



### OFFSPRING SIZE

|OS|AUC|avg. fitness|max|median|
|:---:|:---:|:---:|:---:|:---:|
|  20  | 0.700275 |      43      |  46  |   43   |
|  40  | 0.674613 |     42.7     |  46  |   42   |
|  60  | 0.680942 |     42.7     |  46  |   43   |
|  80  | 0.690477 |      43      |  48  |   44   |
| 100  | 0.65987  |     42.4     |  46  |   42   |

![image-20231128004029359](C:\Users\Hexiang\AppData\Roaming\Typora\typora-user-images\image-20231128004029359.png)

![image-20231128004123465](C:\Users\Hexiang\AppData\Roaming\Typora\typora-user-images\image-20231128004123465.png)



### Final Results
|AUC|avg. fitness|max|median|
|:---:|:---:|:---:|:---:|
|      |              |      |        |