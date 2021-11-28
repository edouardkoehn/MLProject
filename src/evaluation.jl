#Main file for the eval
begin
    using CSV, DataFrames, Plots,MLJ,MLJLinearModels,StatsPlots
    include("utils.jl")
end

begin
    mach_LR_ALL=loadMachine("mach_Log_Classifier_MLJ.jlso")
    println("Log All: " * string(report(mach_LR_ALL).best_history_entry.measurement[1]))
    mach_LR_PUY=loadMachine("mach_Log_Classifier_MLJ_PUY.jlso")
    println("Log PUY " * string(report(mach_LR_PUY).best_history_entry.measurement[1]))
    mach_LR_PUY_STD=loadMachine("mach_Log_Classifier_STD_MLJ_PUY.jlso")
    println("Log All std " * string(report(mach_LR_PUY_STD).best_history_entry.measurement[1]))
end
begin
    #mach_KNN_ALL=loadMachine("mach_KNN_Classifier_MLJ_ALL.jlso")
    #println("KNN All " * string(report(mach_KNN_ALL).best_history_entry.measurement[1]))
    mach_KNN_PUY=loadMachine("mach_KNN_Classifier_MLJ_PUY.jlso")
    println("KNN PUY " * string(report(mach_KNN_PUY).best_history_entry.measurement[1]))
    #mach_KNN_ALL_STD=loadMachine("mach_KNN_Classifier_STD_MLJ_ALL.jlso")
    #println("KNN PUY " * string(report(mach_KNN_PUY).best_history_entry.measurement[1]))
end