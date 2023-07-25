# Temporal Saliency Detection Towards Explainable Transformer-based Timeseries Forecasting

Despite the notable advancements in numerous Transformer-based models, the task of long multi-horizon time series forecasting remains a persistent challenge. 
Venturing beyond the realm of transformers in sequence transduction research, we question the necessity for a technique that can automatically encode saliency-related temporal patterns by establishing connections with appropriate attention heads. 
This paper introduces Temporal Saliency Detection (TSD), an effective approach that builds upon the attention mechanism and applies it to multi-horizon time series prediction. 
While our proposed architecture adheres to the general encoder-decoder structure, it undergoes a significant renovation in the encoder component, wherein we incorporate a series of information contracting and expanding blocks inspired by the U-Net style architecture. 
The TSD approach facilitates the multiresolution analysis of saliency patterns by condensing multi-heads, thereby progressively enhancing the forecasting of complex time series data. 
Empirical evaluations illustrate the superiority of our proposed approach compared to other models across multiple standard benchmark datasets in diverse far-horizon forecasting settings. 
The Initial TSD achieves substantial relative improvements of 31\% and 46\% over several models in the context of multivariate and univariate prediction. 
We believe the comprehensive investigations presented in this study will offer valuable insights and benefits to future research endeavors.
