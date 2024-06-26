### Installation
In order to install the required packages run `pip install -r requirements.txt` or `conda install --file requirements.txt` depending on your prefered installation method. 

### Instructions for creating e.g. Pfo Characterisation SVM model 
Using implementation in [SvmPfoCharacterisationAlgorithm](https://github.com/PandoraPFA/LArContent/blob/master/larpandoracontent/LArTrackShowerId/SvmPfoCharacterisationAlgorithm.cc)

0. Separate your sample of events in two subsets, one for training and one for testing (not necessarily 50% each, but representative of the spectrum of events for your problem).

1. Run over your training subsample with the option 'TrainingSetMode = true'. To do this, edit the corresponding PandoraSettings file (e.g. LArReco/scripts/uboone/PandoraSettings_MicroBooNE_Neutrino.xml) and provide the output training file name like:
```xml
     <algorithm type = "LArSVMClusterCharacterisation">
     ...
     <TrainingSetMode>true</TrainingSetMode>
     <TrainingOutputFileName>OUTPUT_NAME</TrainingOutputFileName>
```
This will create a txt file (note that the .txt will be appended to the OUTPUT_NAME provided) with the features calculated for the events in your input file. You can see an example in the file SVM_training_data_pfocharacterisation_example.txt, containing lines like: 
```05/30/17_16:10:06,98.7135,0.00741252,0.00355123,0.00949093,0.00658028,0.0324565,0.0307609,0.000491683,1``` which are a list of the features starting with a timestamp and finishing with the true value to train for latter classification. In this example, the true value is 1 for a track and 0 for a shower, and the features are the variables computed by the tools:
```xml
    <FeatureTools>
        <tool type = "LArLinearFitFeatureTool"/>
	<tool type = "LArShowerFitFeatureTool"/>
	<tool type = "LArVertexDistanceFeatureTool"/>
    </FeatureTools>
```
which are added in this order: 1) straight line length, 2) mean of difference with straight line, 3) sigma (standard deviation) of difference with straight line, 4) dTdL width, 5) max gap length, 6) RMS of linear fit, 7) shower fit width, 8) vertex distance. Expect the first feature (straight line length) the other ones are normalized divided by the straight line length if the option 'RatioVariables = true'.
The implementation of the tools and available variables can be found in [TrackShowerIdFeatureTool](https://github.com/PandoraPFA/LArContent/blob/master/larpandoracontent/LArTrackShowerId/TrackShowerIdFeatureTool.cc).

*** Note: the next two steps are specific for the rbf (radial basis function) kernel option. For other options, check [scikit-learn](http://scikit-learn.org/stable/)

2. Use the python script rbf_gridsearch_test.py to search for the optimal values of C and gamma for your training data. Edit rbf_gridsearch_test.py and give the text file calculated in step 1 as trainingFile. This script will do a grid search which is time and memory intense, consider sampling your training data accordingly (for example, the SVM_training_data_pfocharacterisation_example.txt contains randomly selected 1000 training examples from the entire training data for this step). At the end, it will report that "The best parameters are C: XXX and gamma: YYY" with a given score. The score is a measurement of the classification, for example in the track-shower characterisation it would be: _ntracks_ * _tracks\_eff_ + _nshowers_ * _showers\_eff_

The python script produces also a plot like [this example one](https://github.com/PandoraPFA/MachineLearningData/blob/master/Example_rbf_output.png), with indicative values of the score in the searched grid. Check this plot to ensure that it is smooth and the selected grid was enough to find a reliable best score (otherwise, if the selected point is at an edge of the grid, consider extending the grid extremes and running again this step).

3. With the values of C and gamma obtained in the previous step, run example.py. Edit it and change C and gamma, and give the appropriate 'trainingFile' name. This step is less time and memory consuming, so the input data can be scaled (for example using 100k training examples). This will give another score, check it against the one in the previous step. If it is very different, it could mean that the sampled training examples used in step 2 were not representative enough of the entire training data, and you might consider running again from step 2 with a larger training sampled input. The output of step 3 will be a .xml file (as well as a .pkl file) with the model, i.e. the SVs, to be used for solving the problem afterwards in your testing data (separated in setp 0), i.e. the input file to be given to the algorithm using it, like in this case:
```xml
    <algorithm type = "LArSvmPfoCharacterisation">
	...
	<SvmFileName>PandoraSvm_PfoCharacterisation_MicroBooNE_mcc7.xml</SvmFileName>
	<SvmName>FinalPfoCharacterisation</SvmName>
	...
    </algorithm>
```

The SvmName is the one added in the output .xml file, and can be changed in [example.py](https://github.com/PandoraPFA/MachineLearningData/blob/master/scripts/example.py)

### Instructions for training a BDT
The general procedure to create a BDT follows the same structure as the SVM discussed above. The training examples produced by Pandora are compatible between the SVM and BDT models as long as they use the same set of `FeatureTools`. An example notebook, ExampleNotebook.ipynb, has been provided to show the procedure for optimising the hyper-parameters, training and validating the BDT. 
