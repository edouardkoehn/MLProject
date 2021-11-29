#Main file for testing the different other Models
begin
    using CSV, DataFrames, Plots,MLJ,MLJLinearModels,StatsPlots
    include("utils.jl")
end
#------------------------------------------------------------------
# Import of the data test
begin
    using CSV, DataFrames, Plots,MLJ,MLJLinearModels,StatsPlots
    test=CSV.read(joinpath(@__DIR__, "..", "data", "testdata.csv"), DataFrame)
    dropmissing!(test)
  
    train=CSV.read(joinpath(@__DIR__, "..", "data", "trainingdata.csv"), DataFrame)
    dropmissing!(train)
    coerce!(train,:precipitation_nextday => Multiclass)
    train=train[shuffle(1:size(train, 1)),:]

    train_PUY= hcat(select(train, r"PUY"), DataFrame(precipitation_nextday=train.precipitation_nextday))
end
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
    randomTree_Classifier=RandomForestClassifier()
    self_DecisionTree=TunedModel(model=randomTree_Classifier,
                                resampling=CV(nfolds=4),
                                tuning=Grid(goal=25),
                                range=range(randomTree_Classifier, :n_trees, lower = 100, upper = 500),
                                measure = auc)
    machine_DecisionTree=machine(self_DecisionTree, select(train, Not(:precipitation_nextday)),train.precipitation_nextday)|>MLJ.fit!
    report(machine_DecisionTree)
end
begin
    machine_RandomForest=machine_DecisionTree
    exportMachine("mach_RandomForest_Classifier.jlso",machine_RandomForest)
end