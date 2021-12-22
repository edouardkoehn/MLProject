### Main file for testing the different other Models
begin
    using CSV, DataFrames, Plots,MLJ,MLJLinearModels,StatsPlots
    include("utils.jl")
end

### Import of the data test
begin
    #import test
    test=importTest()
    #import cleaned train
    train=importTrainCleaned()
    #import filled train
    train_filled=importTrainFilled()
    #subset for PUY weather station  
    train_PUY= hcat(select(train, r"PUY"), DataFrame(precipitation_nextday=train.precipitation_nextday))
    augmented=importAugmented()
end

### KNNClassifier
## KNNClassifier (MLJ) ALL
begin
    using NearestNeighborModels
    KNN_Classifier=KNNClassifier()
    self_KNN_Classifier=TunedModel(model = KNN_Classifier,
            resampling = CV(nfolds = 4),
            tuning = Grid(),
            range=range(KNN_Classifier, :K, values = 1:70),
            measure = auc)
        mach_KNN_Classifier=machine(self_KNN_Classifier, select(train, Not(:precipitation_nextday)),train.precipitation_nextday)|> fit!
        report(mach_KNN_Classifier)
       
end
begin
    exportMachine("mach_KNN_Classifier_MLJ_PUY.jlso", mach_KNN_Classifier_PUY)
end

## KNNClassifier (MLJ) PUY
begin
        mach_KNN_Classifier_PUY=machine(self_KNN_Classifier, select(train_PUY, Not(:precipitation_nextday)),train_PUY.precipitation_nextday)|> fit!
        report(mach_KNN_Classifier_PUY)
end
begin
    exportMachine("mach_KNN_Classifier_MLJ_ALL.jlso", mach_KNN_Classifier_PUY)
end

## KNNClassifier (MLJ)  All Filled
begin
    mach_KNN_Classifier_Filled=machine(self_KNN_Classifier, select(train_filled, Not(:precipitation_nextday)),train_filled.precipitation_nextday)|> fit!
    report(mach_KNN_Classifier_Filled)
end
begin
    exportMachine("mach_KNN_Classifier_MLJ_Filled.jlso", mach_KNN_Classifier_Filled)
end
### Tree base
## XGBoosclassifier ALL
begin
    using MLJXGBoostInterface
	xgb_Classifier = XGBoostClassifier()
    self_XGB = TunedModel(model = xgb_Classifier,
                            resampling = CV(nfolds = 4),
                            tuning = Grid(goal = 25),
                            range = [range(xgb_Classifier, :eta,
                                           lower = 1e-1, upper = .2, scale = :log),#ok
                                     range(xgb_Classifier, :num_round, lower = 600, upper = 800),#ok
                                     range(xgb_Classifier, :max_depth, lower = 5, upper = 7)],#ok
                                     measure = auc)
                 
    mach_XGB= machine(self_XGB, select(train, Not(:precipitation_nextday)),train.precipitation_nextday)|>MLJ.fit!
    report(mach_XGB) 
end
begin
    exportMachine("mach_XGB1_Classifier.jlso",mach_XGB)
end
## XGBBoost with filled data ALL 
begin
    xgb_Classifier_opt = XGBoostClassifier()
    self_XGB_opt = TunedModel(model = xgb_Classifier_opt,
                            resampling = CV(nfolds = 4),
                            tuning = Grid(goal = 25),
                            range = [range(xgb_Classifier_opt, :eta,
                                           lower = 1e-2, upper = .4, scale = :log),#ok
                                     range(xgb_Classifier_opt, :num_round, lower = 60, upper = 600),#ok
                                     range(xgb_Classifier_opt, :max_depth, lower = 3, upper = 10),
                                     range(xgb_Classifier_opt,:subsample, lower=0.5, upper=1),
                                     range(xgb_Classifier_opt, :colsample_bytree, lower=0.5, upper=1)],#ok
                                     measure = auc)
                 
    mach_XGB= machine(self_XG_opt, select(train_filled, Not(:precipitation_nextday)),train_filled.precipitation_nextday)|>MLJ.fit!
    report(mach_XGB) 
end
begin
    exportMachine("mach_XGB1_Classifier_Filled_2.jlso",mach_XGB)
end

## XGBBoost with filled  and standardized data ALL
begin
    train_filled_std = MLJ.transform(fit!(machine(Standardizer(),select(train_filled, Not(:precipitation_nextday)))))
    mach_XGB_3= machine(self_XGB, train_filled_std,train_filled.precipitation_nextday)|>MLJ.fit!
end
begin
    exportMachine("mach_XGB1_Classifier_Filled_Standerdized.jlso",mach_XGB_3)
end

## XGBBoost wiht augmented data
begin
    mach_XGB_4= machine(self_XGB, select(augmented, Not(:precipitation_nextday)),augmented.precipitation_nextday)|>MLJ.fit!
    report(mach_XGB_4).best_model
end
begin
    roc_curve(predict(mach_XGB_4, select(train_filled, Not(:precipitation_nextday))) , train_filled.precipitation_nextday)
    exportMachine("mach_XGB1_Classifier_Augmented.jlso",mach_XGB_4)
end

## DecisionTree classifier ALL
begin
    using MLJDecisionTreeInterface
    decisionTree_Classifier=DecisionTreeClassifier()
    self_DecisionTree=TunedModel(model=decisionTree_Classifier,
                                resampling=CV(nfolds=4),
                                tuning=Grid(goal=25),
                                range=range(decisionTree_Classifier, :max_depth, lower = 2, upper = 6),
                                measure = auc)
    machine_DecisionTree=machine(self_DecisionTree, select(train, Not(:precipitation_nextday)),train.precipitation_nextday)|>MLJ.fit!
    report(machine_DecisionTree)
end
begin
    exportMachine("mach_DecisionTree_Classifier.jlso",machine_DecisionTree)
end

## RandomTree classifier ALL
begin
    randomTree_Classifier=RandomForestClassifier(n_trees=383,max_depth=40,min_samples_split=4)
    self_RandomTree=TunedModel(model=randomTree_Classifier,
                                resampling=CV(nfolds=4),
                                tuning=Grid(goal=25),
                                range=range(randomTree_Classifier, :n_subfeatures, lower = 20, upper = 30),
                                measure = auc)
    machine_RandomForest=machine(self_RandomTree, select(train, Not(:precipitation_nextday)),train.precipitation_nextday)|>MLJ.fit!
    report(machine_RandomForest)
end
begin
    exportMachine("mach_RandomForest_Classifier.jlso",machine_RandomForest)
end

## RandomTree classifier with filled data ALL
begin
    randomTree_Classifier_2=RandomForestClassifier()
    self_RandomTree_2=TunedModel(model=randomTree_Classifier_2,
                                resampling=CV(nfolds=4),
                                tuning=Grid(goal=25),
                                range=range(randomTree_Classifier_2, :n_trees, lower=100,upper=150),
                                measure = auc)

    machine_RandomForest_2=machine(self_RandomTree_2, select(train_filled, Not(:precipitation_nextday)),train_filled.precipitation_nextday)|>MLJ.fit!
    report(machine_RandomForest_2) 
end
begin
    exportMachine("mach_RandomForest_Classifier_Filled.jlso", machine_RandomForest_2)
end
