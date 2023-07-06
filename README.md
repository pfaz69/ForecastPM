# ForecastPM
Code by Paolo Fazzini linked to the article "Forecasting PM levels using machine learning models in polar areas: a comparative study" by Fazzini, Montuori, Pasini, Cuzzucoli, Crotti, Campana, Petracchini and Dobricic. See the article for details.

Tested with Ubuntu 18.04.

Example:
python main.py "FI_5_101983" "esn_2"

To import the related conda environment, use prediction.yml


# Citation
If you find this useful in your research, please cite our paper "Forecasting PM10 Levels Using Machine Learning Models in the Arctic: A Comparative Study". Here's the BibTex reference:

@Article{Fazzini2023pm10,
AUTHOR = {Fazzini, Paolo and Montuori, Marco and Pasini, Antonello and Cuzzucoli, Alice and Crotti, Ilaria and Campana, Emilio Fortunato and Petracchini, Francesco and Dobricic, Srdjan},
TITLE = {Forecasting PM10 Levels Using Machine Learning Models in the Arctic: A Comparative Study},
JOURNAL = {Remote Sensing},
VOLUME = {15},
YEAR = {2023},
NUMBER = {13},
ARTICLE-NUMBER = {3348},
URL = {https://www.mdpi.com/2072-4292/15/13/3348},
ISSN = {2072-4292},
ABSTRACT = {In this study, we present a statistical forecasting framework and assess its efficacy using a range of established machine learning algorithms for predicting Particulate Matter (PM) concentrations in the Arctic, specifically in Pallas (FI), Reykjavik (IS), and Tromso (NO). Our framework leverages historical ground measurements and 24 h predictions from nine models by the Copernicus Atmosphere Monitoring Service (CAMS) to provide PM10 predictions for the following 24 h. Furthermore, we compare the performance of various memory cells based on artificial neural networks (ANN), including recurrent neural networks (RNNs), gated recurrent units (GRUs), long short-term memory networks (LSTMs), echo state networks (ESNs), and windowed multilayer perceptrons (MLPs). Regardless of the type of memory cell chosen, our results consistently show that the proposed framework outperforms the CAMS models in terms of mean squared error (MSE), with average improvements ranging from 25% to 40%. Furthermore, we examine the impact of outliers on the overall performance of the model.},
DOI = {10.3390/rs15133348}
}


