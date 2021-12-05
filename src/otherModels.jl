#Main file for testing the different other Models
begin
    using CSV, DataFrames, Plots,MLJ,MLJLinearModels,StatsPlots
    include("utils.jl")
end
#------------------------------------------------------------------
# Import of the data test
begin
    test=CSV.read(joinpath(@__DIR__, "..", "data", "testdata.csv"), DataFrame)
    dropmissing!(test)
  
    train=CSV.read(joinpath(@__DIR__, "..", "data", "trainingdata.csv"), DataFrame)
    coerce!(train,:precipitation_nextday => Multiclass)
    train=train[shuffle(1:size(train, 1)),:]

    train_filled= MLJ.transform(fit!(machine(FillImputer(), train)), train)
    coerce!(train_filled,:precipitation_nextday => Multiclass)
    train_filled=train_filled[shuffle(1:size(train_filled, 1)),:]
    
    dropmissing!(train)
    
    train_PUY= hcat(select(train, r"PUY"), DataFrame(precipitation_nextday=train.precipitation_nextday))
end
#KNN
#------------------------------------------------------------------
#KNNClassifier (MLJ) ALL
begin
    using NearestNeighborModels
    KNN_Classifier=KNNClassifier()
    self_KNN_Classifier=TunedModel(model = KNN_Classifier,
            resampling = CV(nfolds = 10),
            tuning = Grid(),
            range=range(KNN_Classifier, :K, values = 1:70),
            measure = auc)
        mach_KNN_Classifier=machine(self_KNN_Classifier, select(train, Not(:precipitation_nextday)),train.precipitation_nextday)|> fit!
        report(mach_KNN_Classifier).best_model
        evaluate!(mach_KNN_Classifier, measure=auc)
end
begin
    exportMachine("mach_KNN_Classifier_MLJ_PUY.jlso", mach_KNN_Classifier_PUY)
end

#------------------------------------------------------------------
#KNNClassifier (MLJ) PUY
begin
        mach_KNN_Classifier_PUY=machine(self_KNN_Classifier, select(train_PUY, Not(:precipitation_nextday)),train_PUY.precipitation_nextday)|> fit!
        report(mach_KNN_Classifier_PUY).best_model
        evaluate!(mach_KNN_Classifier_PUY, measure=auc)
end
begin
    exportMachine("mach_KNN_Classifier_MLJ_ALL.jlso", mach_KNN_Classifier_PUY)
end

#--------------------------------------------------------------

#Tree based
#-------------------------------------------------------------------
# XGBoosclassifier
begin
    using MLJXGBoostInterface
	xgb_Classifier = XGBoostClassifier()
    self_XGB = TunedModel(model = xgb_Classifier,
                            resampling = CV(nfolds = 4),
                            tuning = Grid(goal = 25),
                            range = [range(xgb_Classifier, :eta,
                                           lower = 1e-2, upper = .3, scale = :log),#ok
                                     range(xgb_Classifier, :num_round, lower = 100, upper = 700),#ok
                                     range(xgb_Classifier, :max_depth, lower = 1, upper = 4)],#ok
                                     measure = auc)
                 
    mach_XGB= machine(self_XGB, select(train, Not(:precipitation_nextday)),train.precipitation_nextday)|>MLJ.fit!
    report(mach_XGB) 
end
begin
    exportMachine("mach_XGB1_Classifier.jlso",mach_XGB)
end
#-------------------------------------------------------------------
# XGBBoost with filled data
begin
    mach_XGB_2= machine(self_XGB, select(train_filled, Not(:precipitation_nextday)),train_filled.precipitation_nextday)|>MLJ.fit!
    report(mach_XGB_2)
end

#-------------------------------------------------------------------
# DecisionTree classifier
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
#-------------------------------------------------------------------
# RandomTree classifier
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
#-------------------------------------------------------------------
# RandomTree classifier with filled data
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
#-------------------------------------------------------------------
