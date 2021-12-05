#Main file for the data exploration and Visualisation
# Import of the data test
begin
    using CSV, DataFrames, Plots,MLJ,MLJLinearModels,StatsPlots
    test=CSV.read(joinpath(@__DIR__, "..", "data", "testdata.csv"), DataFrame)
    println("Shape of the test set :",size(test))
    dropmissing!(test)
    println("Shape of the test set cleaned :",size(test))

    train=CSV.read(joinpath(@__DIR__, "..", "data", "trainingdata.csv"), DataFrame)
    println("Shape of the training set :" ,size(train))
    coerce!(train,:precipitation_nextday => Multiclass)

    train_cleaned=dropmissing(train)
    println("Shape of the training set cleaned :" ,size(train_cleaned))
    coerce!(train_cleaned,:precipitation_nextday => Multiclass)

    train_filled= MLJ.transform(fit!(machine(FillImputer(), train)), train)
    println("Shape of the training set cleaned :" ,size(train_filled))
    coerce!(train_filled,:precipitation_nextday => Multiclass)

end
# data set statistics
begin
    train_info =describe(train[:,r"PUY"],:mean, :min, :median, :max ,:std, :nmissing)
    test_info  =describe(test[:,r"PUY"],:mean, :min, :median, :max ,:std, :nmissing)
    filled_info =describe(train_filled[:,r"PUY"],:mean, :min, :median, :max ,:std, :nmissing)

    CSV.write("data/train_PUY_info.csv", train_info)
    CSV.write("data/test_PUY_info.csv", test_info)
    CSV.write("data/filled_PUY_info.csv", filled_info)
    @df train_cleaned boxplot([:PUY_air_temp_1 :PUY_air_temp_2 :PUY_air_temp_3 :PUY_air_temp_4],yaxis="[C]",Layout=(boxmode="group"))
    subplot = twinx()
    @df train_cleaned boxplot!(subplot,[:PUY_delta_pressure_1 :PUY_delta_pressure_2 :PUY_delta_pressure_3 :PUY_delta_pressure_4], yaxis="[hpa]", size= (800, 500),Layout=(boxmode="group"))
end

#Subset data for pully
begin
    stationPully = train_cleaned[:,r"PUY"]
    print(names(stationPully))
end

#Plots temp plot
begin
    moving_average(vs,n) = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]
    temp_1=moving_average(stationPully.PUY_air_temp_1,100)
    temp_2=moving_average(stationPully.PUY_air_temp_2,100)
    temp_3=moving_average(stationPully.PUY_air_temp_3,100)
    temp_4=moving_average(stationPully.PUY_air_temp_4,100)
    p_temp=plot([temp_1 temp_2 temp_3 temp_4], ylabel="Avg T[C]", title="Temperature" , legend=false)
end
#Pressure drop plot
begin
    pressure_1=moving_average(stationPully.PUY_delta_pressure_1,100)
    pressure_2=moving_average(stationPully.PUY_delta_pressure_2,100)
    pressure_3=moving_average(stationPully.PUY_delta_pressure_3,100)
    pressure_4=moving_average(stationPully.PUY_delta_pressure_4,100)
    p_pressure=plot([pressure_1 pressure_2 pressure_3 pressure_4], ylabel=" Delta P[hpa]",xlabel="Days", title="Pressure",legend=false)
end
#Pressure drop plot
begin
    sun_1=moving_average(stationPully.PUY_sunshine_1,100)
    sun_2=moving_average(stationPully.PUY_sunshine_2,100)
    sun_3=moving_average(stationPully.PUY_sunshine_3,100)
    sun_4=moving_average(stationPully.PUY_sunshine_4,100)
    p_sun=plot([sun_1 sun_2 sun_3 sun_4], ylabel=" Sun [min]", title="Sun", label=["Morning" "MID Day" "Evening" "Night"], legend=:bottomright)
end

#Corrolation plot betwenn the predictores
begin
    using StatsPlots
    data_avg= DataFrame( T_1= temp_1,T_2= temp_2,T_3= temp_3,T_4= temp_4, Sun_1=sun_1, Sun_2=sun_2, Sun_3=sun_3, Sun_4=sun_4, P_1=pressure_1, P_2=pressure_2, P_3=pressure_3, P_4=pressure_4 )
    p_corr=@df data_avg corrplot([:T_1 :Sun_1 :P_1 :T_2])
    savefig(p_corr, "figures/Plot_corr_AVG_data.pdf")
end

#Complete figure
begin
    savefig(plot(p_temp, p_pressure,p_sun, layout=(3,1), size=(700,1000)), "figures/Plot_AVG_data.pdf")
end

# Perform a PCA
begin
    using MLJMultivariateStatsInterface, Statistics, LinearAlgebra,MultivariateStats, MLJ
   
    pca_train= transpose(Array(select(train_cleaned, Not(:precipitation_nextday))))
    pca_label=train_cleaned.precipitation_nextday
    
    M = fit!(machine(MLJMultivariateStatsInterface.PCA(pratio=1, maxoutdim=4), pca_train))
    pca_train_transformed =MLJ.transform(M, pca_train)

    plot(pca_train_transformed.x1, pca_train_transformed.x2, seriestype=:scatter, label="")
end
# Correlation between the predictors
begin
    corMatrix=cor(Matrix(select(train_cleaned, Not(:precipitation_nextday))))
    corMatrix[1:end .==1]
    corrplot(M)
    df_corMatrix=DataFrame(corMatrix, :auto)

    df_corMatrix[df_corMatrix .==1]
    print(findall(x->findall(x->x==1, corMatrix))

end