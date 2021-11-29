#Main file for the eval
begin
    using CSV, DataFrames, Plots,MLJ,MLJLinearModels,StatsPlots
    include("utils.jl")
    test=CSV.read(joinpath(@__DIR__, "..", "data", "testdata.csv"), DataFrame)
    dropmissing!(test)
end
#Logistic Classifier
begin
    mach_LR_ALL=loadMachine("mach_Log_Classifier_MLJ_ALL.jlso")
    println("Log All: " * string(report(mach_LR_ALL).best_history_entry.measurement[1]))
    mach_LR_PUY=loadMachine("mach_Log_Classifier_MLJ_PUY.jlso")
    println("Log PUY " * string(report(mach_LR_PUY).best_history_entry.measurement[1]))
    mach_LR_ALL_STD=loadMachine("mach_Log_Classifier_STD_MLJ_ALL.jlso")
    println("Log All std " * string(report(mach_LR_ALL_STD).best_history_entry.measurement[1]))
end
#plotting classifier
begin
    p_LR_All=plotMachine(mach_LR_ALL, "LR_ALL")
    p_LR_PUY=plotMachine(mach_LR_PUY, "LR_PUY")
    p_LR_All_STD=plotMachine(mach_LR_ALL_STD,"LR_ALL_STD")
    plot(p_LR_All,p_LR_All_STD,p_LR_PUY)
end
#KNN Classifier
begin
    mach_KNN_ALL=loadMachine("mach_KNN_Classifier_MLJ_ALL.jlso")
    println("KNN All " * string(report(mach_KNN_ALL).best_history_entry.measurement[1]))
    mach_KNN_PUY=loadMachine("mach_KNN_Classifier_MLJ_PUY.jlso")
    println("KNN PUY " * string(report(mach_KNN_PUY).best_history_entry.measurement[1]))
    #mach_KNN_ALL_STD=loadMachine("mach_KNN_Classifier_STD_MLJ_ALL.jlso")
    #println("KNN PUY " * string(report(mach_KNN_PUY).best_history_entry.measurement[1]))
end
#plotting KNN
begin
    p_KNN_All=plotMachine(mach_KNN_ALL, "KNN_ALL")
    p_KNN_PUY=plotMachine(mach_KNN_PUY, "KNN_PUY")
    plot(p_KNN_All,p_KNN_PUY)
end

#Treebased classifier
begin
    mach_XGBBoost=loadMachine("mach_XGB_Classifier.jlso")
    println("XBGBoost "* string(report(mach_XGBBoost).best_history_entry.measurement[1]))

    mach_XGBBoost1=loadMachine("mach_XGB1_Classifier.jlso")
    println("XBGBoost1 "* string(report(mach_XGBBoost1).best_history_entry.measurement[1]))

    mach_DecisionTree=loadMachine("mach_DecisionTree_Classifier.jlso")
    println("DecisionTree "* string(report(mach_DecisionTree).best_history_entry.measurement[1]))

    mach_RandomForest=loadMachine("mach_RandomForest_Classifier.jlso")
    println("RandomForest "* string(report(mach_RandomForest).best_history_entry.measurement[1]))

end

begin
    p_Random=plotMachine(mach_RandomForest, "RandomForest")
    report(mach_XGBBoost).best_model
    p_XGB1=plotMachine(mach_XGBBoost1, "XGBBoost")
end