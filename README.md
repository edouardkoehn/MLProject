# MLProject

## Organisation of the repository
This repo contained all the script and filles used in the project of BIO-322. All the code has been written in Julia (https://julialang.org/). You would find the report of the project at the root of the repo. This repository is composed of four folders : 
- data : 
It contains all the data needed to run the code. 
- src:
It contains the julia scripts that were used for generating the model, the submission, the data explorations and the evaluation of the models
- models: 
It contains all the generated models during the process. The files are in jlso format.
- submission:
It contains all the submission uploaded on Kaggle

## Wokflow
For this project, we began to do a data exploration with the script `dataVisualisation.jl`. Then we generated several models with the following scripts : `linearModels.jl`, `otherModels.jl` and `neuronalModels.jl`. We did the models evaluation in an other scripts :`evaluation.jl`.
Finally, we generated a submission with the script `submission.jl`

Edouard Koehn
