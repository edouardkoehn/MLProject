### Main file for the evaluation
# In this file, we simply print the performance of each model and plot some comparison between the models

begin
    using CSV, DataFrames, Plots,MLJ,MLJLinearModels,StatsPlots,MLJXGBoostInterface,MLJDecisionTreeInterface
    include("utils.jl")
end
begin
    test=importTest()
    train_cleaned=importTrainCleaned()
    train_PUY= hcat(select(train_cleaned, r"PUY"), DataFrame(precipitation_nextday=train_cleaned.precipitation_nextday))
    train_filled=importTrainFilled()
end

##Logistic Classifier
begin
    #Import or create the different model
    #Logistic Classifier with cleaned data
    mach_LR_NoR_results=evaluate(LogisticClassifier(penalty=:none), select(train_cleaned, Not(:precipitation_nextday)), train_cleaned.precipitation_nextday, measure=auc, resampling=CV(nfolds = 10))
    mach_LR_L1=loadMachine("mach_Log_Classifier_MLJ_L1.jlso")
    mach_LR_L2=loadMachine("mach_Log_Classifier_MLJ_L2.jlso")
    #Logistic Classifier with filled data
    mach_LR_L2_Filled=loadMachine("mach_Log_Classifier_MLJ_Filled_L2.jlso")
    #Logistic Classifier with standardized data
    mach_LR_L2_Std=loadMachine("mach_Log_Classifier_STD_MLJ_ALL_STD.jlso")
    #LogisticClassifier with PUY station data
    mach_LR_NoR_PUY_results=evaluate(LogisticClassifier(penalty=:none), select(train_PUY, Not(:precipitation_nextday)), train_PUY.precipitation_nextday, measure=auc, resampling=CV(nfolds = 10))
    mach_LR_L1_PUY=loadMachine("mach_Log_Classifier_MLJ_PUY_L1.jlso")
    mach_LR_L2_PUY=loadMachine("mach_Log_Classifier_MLJ_PUY_L2.jlso")
    report(mach_LR_L2_Filled).best_model
end
begin
    #Logistic Classifier with cleaned data
    println("LR NoR, AUC:" * string(mach_LR_NoR_results.measurement[1])) 
    println("LR L1, AUC:" * string(getMeasurements(mach_LR_L1)))
    println("LR L2, AUC:" * string(getMeasurements(mach_LR_L2))) 
    #Logistic Classifier with filled data
    println("LR L2 Filed data, AUC:" * string(getMeasurements(mach_LR_L2_Filled)))
    #Logistic Classifier with standardized data
    println("LR L2 standardized data, AUC:" * string(getMeasurements(mach_LR_L2_Std)))
    #LogisticClassifier with PUY station data
    println("LR NoR PUY, AUC: " * string(mach_LR_NoR_PUY_results.measurement[1]))
    println("LR L1 PUY, AUC:" * string(getMeasurements(mach_LR_L1_PUY)))
    println("LR L2 PUY, AUC:" * string(getMeasurements( mach_LR_L2_PUY)))   
end
begin
    #Plot the results for logistic classifier
    labels=["No Regularisation", "L1 Regularisation" ,"L2 Regularisation" ,"L2 Regularisation, filled data" ,"L2 Regularisation, standardized data" ,"No Regularisation, PUY", "L1 Regularisation, PUY", "L2 Regularisation, PUY"]
    bar([mach_LR_NoR_results.measurement[1],getMeasurements(mach_LR_L1),getMeasurements(mach_LR_L2),getMeasurements(mach_LR_L2_Filled), getMeasurements(mach_LR_L2_Std), mach_LR_NoR_PUY_results.measurement[1],getMeasurements(mach_LR_L1_PUY),getMeasurements( mach_LR_L2_PUY)],
     xticks=(1:10, labels), label="AUC",xrotation = 45, title="Logistic classifier comparison", figsze=(500,400),tickfontsize=5)
    ylims!(0.5,0.92)
end
begin
    #Plot the learning curve with the L2 Regularisation
    scatter(report(mach_LR_L2).plotting.parameter_values[:,1] , report(mach_LR_L2).plotting.measurements[:,1], x_scale=:log, label="L2 Regularization,the cleaned data")
    scatter!(report(mach_LR_L2_Filled).plotting.parameter_values[:,1] , report(mach_LR_L2_Filled).plotting.measurements[:,1], x_scale=:log, label="L2 Regularization,the filled data", title="Learning curve", legend=:bottomright)
end
##KNN Classifier
begin
    #KNN classifier on all the data
    mach_KNN_ALL=loadMachine("mach_KNN_Classifier_MLJ_ALL.jlso")
    println("KNN All, AUC: " * string(getMeasurements(mach_KNN_ALL)))
    #KNN classifier on all the data
    mach_KNN_Filled=loadMachine("mach_KNN_Classifier_MLJ_Filled.jlso")
    println("KNN Filled, AUC: " * string(getMeasurements(mach_KNN_Filled)))
    #KNN classifier the subset of PUY weather station
    mach_KNN_PUY=loadMachine("mach_KNN_Classifier_MLJ_PUY.jlso")
    println("KNN PUY, AUC: " * string(getMeasurements(mach_KNN_PUY)))
    report(mach_KNN_Filled).best_model
end

##Treebased classifier
begin
    #First implementation of XGBboost
    mach_XGBBoost=loadMachine("mach_XGB_Classifier.jlso")
    println("XBGBoost, AUC: "*  string(getMeasurements(mach_XGBBoost)))
    #Optimised version of XGBboost
    mach_XGBBoost1=loadMachine("mach_XGB1_Classifier.jlso")
    println("XBGBoost1, AUC: "*  string(getMeasurements(mach_XGBBoost1)))
    #Optimised version of XGBboost with filled data
    mach_XGBBoost1_Filled=loadMachine("mach_XGB1_Classifier_Filled.jlso")
    println("XBGBoost1 Filled data, AUC: "*  string(getMeasurements(mach_XGBBoost1_Filled)))
    #Optimised version of XGBboost with filled and standardized data
    mach_XGBBoost1_Filled_standerdized=loadMachine("mach_XGB1_Classifier_Filled_Standerdized.jlso")
    println("XBGBoost1 Filled and Standerdized data, AUC: "*  string(getMeasurements(mach_XGBBoost1_Filled_standerdized)))
    #Optimised version of XGBboost with augmented data
    mach_XGBBoost1_Aug=loadMachine("mach_XGB1_Classifier_augmented.jlso")
    println("XBGBoost1 Augmented, AUC: "*  string(getMeasurements(mach_XGBBoost1_Aug)))
    #Decision Tree implementation
    mach_DecisionTree=loadMachine("mach_DecisionTree_Classifier.jlso")
    println("DecisionTree, AUC: "* string(getMeasurements(mach_DecisionTree)))
    #Random Forest implementation
    mach_RandomForest=loadMachine("mach_RandomForest_Classifier.jlso")
    println("RandomForest, AUC: "* string(getMeasurements(mach_RandomForest)))
    #Random Forest implementation
    mach_RandomForest_filled=loadMachine("mach_RandomForest_Classifier_Filled.jlso")
    println("RandomForest Filled data, AUC: "* string(getMeasurements(mach_RandomForest_filled)))
end

begin
    #Plot the results for the tree based classifier
    labels=["XGBoost", "XGBBoost (filled)" ,"XGBBoost (augmented)", "DecisionTree" ,"RandomForest" ]
    bar([getMeasurements(mach_XGBBoost1),getMeasurements(mach_XGBBoost1_Filled),getMeasurements(mach_XGBBoost1_Aug),getMeasurements(mach_DecisionTree), getMeasurements(mach_RandomForest)],
     xticks=(1:10, labels), label="AUC",xrotation = 45, title="Logistic classifier comparison", figsze=(500,400),tickfontsize=5)
    ylims!(0.8,0.95)
end

begin
    #Plot the CV results for the two best model of XGBBoost
    scatter([1:5],report(mach_XGBBoost1_Filled).best_history_entry.per_fold, label="XGB, filled data")
    scatter!([1:5],report(mach_XGBBoost1_Aug).best_history_entry.per_fold, label="XGB, augmented data", title="Measurement of the CV")
    ylims!(0.9,0.95)
end

## NN classifier
begin
    #First implementation of NN network
    mach_NN_1=loadMachine("mach_NN_classifier_n=5.jlso")
    println("NN_1 "* string(getMeasurements(mach_NN_1)))
    #Second implementation of NN network
    mach_NN_2=loadMachine("mach_NN_classifier_n=200.jlso")
    println("NN_2 "* string(getMeasurements(mach_NN_2)))
    #Optimised implementation of NN network
    #The evaluation of the optimsed NN network is done in the  neuronalModels.jl script L60
end

