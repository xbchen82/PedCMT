## Structure info
Updating: 2023-06-16<br>

URL:[（click here）](https://github.com/sellenzh/pedCMT)<br>
structure info：

```
├── checkpoints                 <- train model save dir。
│   ├── JAAD_all.pt             
│   ├── JAAD_beh.pt
│   └── PIE.pt
│
├── logs                        <- train logs save dir
│   └── PIE                    
│       └── ...                 
│
├── PIE                         <- dataset（download:[PIE]）
│   └── ...                     <- note: need to unzip `annotations.zip`,`annotations_vehicle.zip`,
│          ├── ...                 <- `annotations_attributes.zip`
│          └── ....                <- 
├─── JAAD...  
│       ├── ...                 <- [JAAD]
│       └── ...                 <- ...
│
├── utils                       
│   ├── pie_data.py             
│   └── pie_preprocessing.py    
│
├── model                       <- models save dir
│   ├── BottleNeck.py          
│   ├── FFN.py                  
│   ├── model_blocks.py         
│   └── main_model.py           
│
├── pie.py                      
│                                 
├── jaad.py                     
│                                 
└── README.md
```
Download :[PIE](https://github.com/aras62/PIE.git)<br>
[JAAD](https://github.com/ykotseruba/JAAD.git)
