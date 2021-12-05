#Main file for the eval
begin
    using CSV, DataFrames, Plots,MLJ,MLJLinearModels,StatsPlots,MLJXGBoostInterface,MLJDecisionTreeInterface
    include("utils.jl")
    test=CSV.read(joinpath(@__DIR__, "..", "data", "testdata.csv"), DataFrame)

    train=CSV.read(joinpath(@__DIR__, "..", "data", "trainingdata.csv"), DataFrame)
    dropmissing!(train)
    coerce!(train,:precipitation_nextday => Multiclass)
    train=train[shuffle(1:size(train, 1)),:]

    train_PUY= hcat(select(train, r"PUY"), DataFrame(precipitation_nextday=train.precipitation_nextday))
    dropmissing!(test)
end
#Logistic Classifier
begin
    mach_LR_NoR=loadMachine("mach_Log_Classifier_MLJ_NoR.jlso")
    println("LR NoR: " * string(mean(CV_Auc(mach_LR_NoR, train, 10))))
    mach_LR_L1=loadMachine("mach_Log_Classifier_MLJ_L1.jlso")
    evaluate!(mach_LR_L1, measure=auc)
    mean(CV_Auc(mach_LR_L1, train, 10))
    println("LR L1 " * string(mean(CV_Auc(mach_LR_L1, train, 10))))
    mach_LR_L2=loadMachine("mach_Log_Classifier_MLJ_L2.jlso")
    println("LR L2 " * string(mean(CV_Auc(mach_LR_L2, train, 10))))
    
end

#KNN Classifier
begin
    mach_KNN_ALL=loadMachine("mach_KNN_Classifier_MLJ_ALL.jlso")
    println("KNN All " * string(mean(CV_Auc(mach_KNN_ALL, train, 10))))
    mach_KNN_PUY=loadMachine("mach_KNN_Classifier_MLJ_PUY.jlso")
    println("KNN PUY " * string(mean(CV_Auc(mach_KNN_PUY, train_PUY, 10))))
end

#Treebased classifier
begin
    mach_XGBBoost=loadMachine("mach_XGB_Classifier.jlso")
    predict(mach_XGBBoost, select(train, Not(:precipitation_nextday)))
    println("XBGBoost "* string(mean(CV_Auc(mach_XGBBoost, train, 10))))
    mach_XGBBoost1=loadMachine("mach_XGB1_Classifier.jlso")
    println("XBGBoost1 "* string(mean(CV_Auc(mach_XGBBoost1, train, 10))))
    mach_DecisionTree=loadMachine("mach_DecisionTree_Classifier.jlso")
    println("DecisionTree "* string(mean(CV_Auc(mach_DecisionTree, train, 10))))
    mach_RandomForest=loadMachine("mach_RandomForest_Classifier.jlso")
    println("RandomForest "* string(mean(CV_Auc(mach_RandomForest, train, 10))))
    report(mach_RandomForest)
end
## NN classifier
begin
    mach_NN_1=loadMachine("mach_NN_classifier_n=5.jlso")
    println("NN_1 "* string(mean(CV_Auc(mach_NN_1, train, 10))))
    mach_NN_2=loadMachine("mach_NN_classifier_n=200.jlso")
    println("NN_2 "* string(mean(CV_Auc(mach_NN_2, train, 10))))
    mach_NN_opt=loadMachine("mach_NN_optimised.jlso")
    println("NN_Opt "* string(mean(CV_Auc(mach_NN_opt, train, 10))))
    
end

#plotting classifier
begin
    p_LR_All=plotMachine(mach_LR_ALL, "LR_ALL")
    p_LR_PUY=plotMachine(mach_LR_PUY, "LR_PUY")
    p_LR_All_STD=plotMachine(mach_LR_ALL_STD,"LR_ALL_STD")
    plot(p_LR_All,p_LR_All_STD,p_LR_PUY)
end

#plotting KNN
begin
    p_KNN_All=plotMachine(mach_KNN_ALL, "KNN_ALL")
    p_KNN_PUY=plotMachine(mach_KNN_PUY, "KNN_PUY")
    plot(p_KNN_All,p_KNN_PUY)
end

begin
    p_Random=plotMachine(mach_RandomForest, "RandomForest")
    report(mach_RandomForest).best_model
    p_XGB1=plotMachine(mach_XGBBoost1, "XGBBoost")
end