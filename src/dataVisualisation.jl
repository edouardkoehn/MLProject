### Main file for the data exploration and Visualisation
begin
    using CSV, DataFrames, Plots,MLJ,MLJLinearModels,StatsPlots, Statistics
    include("utils.jl")
end
## Import the data 
begin
    # test  data
    test=importTest()
    # raw training data
    train=CSV.read(joinpath(@__DIR__, "..", "data", "trainingdata.csv"), DataFrame)
    println("Shape of the training set :" ,size(train))
    coerce!(train,:precipitation_nextday => Multiclass)

    # cleaned training data
    train_cleaned=importTrainCleaned()

    # filled training data
    train_filled= importTrainFilled()
end

## Data set statistics
begin
    #get the general data set statistics
    train_info =describe(train,:mean, :min, :median, :max ,:std, :nmissing)
    train_cleaned=describe(train_cleaned,:min, :median, :max ,:std, :nmissing)
    test_info  =describe(test,:mean, :min, :median, :max ,:std, :nmissing)
    filled_info =describe(train_filled,:mean, :min, :median, :max ,:std, :nmissing)
end
begin
    #Plot the distribution of missing data in the dataset
    boxplot(train_info.nmissing ./3176,orientation=:h, label="% NaN", title="Boxplot of the percent of missing data", xlabel="%",yaxis=false, yticks=false)
end

## General plot on the test and the training sets
begin
    plot(train_cleaned.PUY_air_temp_1 ,yaxis="[C]",label="Train_PUY_T_1", xlim=([1:1000]))
    plot!(test.PUY_air_temp_1,xlims=(1,100),label="Test_PUY_T_1", title="Temperature Pully 1")
end

## Analyze the subset of Pully weather station
begin
    stationPully = train_cleaned[:,r"PUY"]
    print(names(stationPully))
end
begin
    #Temperature plot
    moving_average(vs,n) = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]
    temp_1=moving_average(stationPully.PUY_air_temp_1,100)
    temp_2=moving_average(stationPully.PUY_air_temp_2,100)
    temp_3=moving_average(stationPully.PUY_air_temp_3,100)
    temp_4=moving_average(stationPully.PUY_air_temp_4,100)
    p_temp=plot([temp_1 temp_2 temp_3 temp_4], ylabel="Avg T[C]", title="Temperature" , label=[:"PUY_1" :"PUY_2" :"PUY_3" :"PUY_4"],legend=true)
end
begin
    #Pressure drop plot
    pressure_1=moving_average(stationPully.PUY_delta_pressure_1,100)
    pressure_2=moving_average(stationPully.PUY_delta_pressure_2,100)
    pressure_3=moving_average(stationPully.PUY_delta_pressure_3,100)
    pressure_4=moving_average(stationPully.PUY_delta_pressure_4,100)
    p_pressure=plot([pressure_1 pressure_2 pressure_3 pressure_4], ylabel=" Delta P[hpa]",xlabel="Days", title="Drop of Pressure",label=[:"PUY_1" :"PUY_2" :"PUY_3" :"PUY_4"],legend=true)
end
begin
    #Sunshine plot
    sun_1=moving_average(stationPully.PUY_sunshine_1,100)
    sun_2=moving_average(stationPully.PUY_sunshine_2,100)
    sun_3=moving_average(stationPully.PUY_sunshine_3,100)
    sun_4=moving_average(stationPully.PUY_sunshine_4,100)
    p_sun=plot([sun_1 sun_2 sun_3 sun_4], ylabel=" Sun [min]", title="Sun", label=["Morning" "MID Day" "Evening" "Night"], legend=:bottomright)
end

## Corrolation betwenn the predictors of the subset
begin
    #Corrolation plots between the predicors
    data_avg= DataFrame( T_1= temp_1,T_2= temp_2,T_3= temp_3,T_4= temp_4, Sun_1=sun_1, Sun_2=sun_2, Sun_3=sun_3, Sun_4=sun_4, P_1=pressure_1, P_2=pressure_2, P_3=pressure_3, P_4=pressure_4 )
    p_corr=@df data_avg corrplot([:T_1 :Sun_1 :P_1 :T_2])
end
begin
    # Correlation Matrix on the predictors
    corMatrix=broadcast(abs,cor(Matrix(select(train_cleaned, Not(:precipitation_nextday)))))
    
    heatmap(1:size(corMatrix)[1],1:size(corMatrix)[2], (corMatrix[1:100,1:100]),c=cgrad([:blue, :white,:red, :yellow]),xlabel="Predictors", ylabel="Predictors",title="Corrolation between the classifier", figsize=(1000,1000))
    heatmap(1:100,1:100, (corMatrix[1:100,1:100]),c=cgrad([:blue, :white,:red, :yellow]),xlabel="Predictors", ylabel="Predictors",title="Corrolation between the classifier", figsize=(1000,1000))
    heatmap(1:50,1:50, (corMatrix[1:50,1:50]),c=cgrad([:blue, :white,:red, :yellow]),xlabel="Predictors", ylabel="Predictors",title="Corrolation between the classifier", figsize=(1000,1000))

end

## Perform a PCA
begin
    using MLJMultivariateStatsInterface, MultivariateStats
   
    pca_train= (Array(select(train_cleaned, Not(:precipitation_nextday))))
    pca_label=train_cleaned.precipitation_nextday
    
    M = fit!(machine(MLJMultivariateStatsInterface.PCA( pratio=1, maxoutdim=4), pca_train))
    pca_train_transformed =MLJ.transform(M, pca_train)
    (pca_train)
    scatter(pca_train_transformed.x2, pca_train_transformed.x3,  mc = [:navy, :crimson],label=pca_label,xlabel="PCA1", ylabel="PCA2")
end
