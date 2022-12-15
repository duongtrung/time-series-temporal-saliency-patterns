# time-series-temporal-saliency-patterns
The implementation of the paper: Put Attention to Temporal Saliency Patterns of Multi-Horizon Time Series

Time series, sets of sequences in chronological order, are essential data in statistical research with many forecasting applications. 
Although recent performance in many Transformer-based models has been noticeable, long multi-horizon time series forecasting remains a very challenging task.
Going beyond transformers in sequence translation and transduction research, we observe the effects of down-and-up samplings that can nudge temporal saliency patterns to emerge in time sequences. 
Motivated by the mentioned observation, in this paper, we propose a novel architecture, Temporal Saliency Detection (TSD), on top of the attention mechanism and apply it to multi-horizon time series prediction.
We renovate the traditional encoder-decoder architecture by making as a series of deep convolutional blocks to work in tandem with the multi-head self-attention. 
The proposed TSD approach facilitates the multiresolution of saliency patterns upon condensed multi-heads, thus progressively enhancing complex time series forecasting.
Experimental results illustrate that our proposed approach has significantly outperformed existing state-of-the-art methods across multiple standard benchmark datasets in many far-horizon forecasting settings.
Overall, TSD achieves 31% and 46% relative improvement over the current state-of-the-art models in multivariate and univariate time series forecasting scenarios on standard benchmarks.
