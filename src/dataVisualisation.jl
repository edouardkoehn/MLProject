### Main file for the data exploration and Visualisation
begin
    using CSV, DataFrames, Plots,MLJ,MLJLinearModels,StatsPlots, Statistics
    include("utils.jl")
end

### Import the data 
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
    # Standardizer and cleaned training data
    train_std = MLJ.transform(fit!(machine(Standardizer(),select(train_cleaned, Not(:precipitation_nextday)))))
   
end

### Data set statistics
begin
    #get the general data set statistics
    train_info =describe(train,:mean, :min, :median, :max ,:std, :nmissing)
    train_cleaned_info=describe(train_cleaned,:min, :median, :max ,:std, :nmissing)
    test_info  =describe(test,:mean, :min, :median, :max ,:std, :nmissing)
    filled_info =describe(train_filled,:mean, :min, :median, :max ,:std, :nmissing)
    #Export the statistics of each dataset into data
    CSV.write(joinpath(@__DIR__, "..", "data", "train_info.csv"), train_info)
    CSV.write(joinpath(@__DIR__, "..", "data", "train_cleaned_info.csv"), train_cleaned_info)
    CSV.write(joinpath(@__DIR__, "..", "data", "filled_info.csv"), filled_info)
    #Print the percentage of missing, true and false in the training set
    println("Train:: %missing: ", 100*(size(train_cleaned)[1]/(size(train)[1])))
    println("Train:: %True: " ,  100*(count(train_cleaned.precipitation_nextday .==true)/size(train_cleaned)[1]))
    println("Train:: %False: ", 100*(count(train_cleaned.precipitation_nextday .==false)/size(train_cleaned)[1]))

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

### Analyze the subset of Pully weather station
# We used a moving average filter to reduce the noise of the data and to make the plot more comprensible
begin
    stationPully = train_cleaned[:,r"PUY"]
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

### Corrolation betwenn the predictors
begin
    #Corrolation plots between the predicors of the Pully station
    data_avg= DataFrame(T_1= temp_1,T_2= temp_2,T_3= temp_3,T_4= temp_4, Sun_1=sun_1, Sun_2=sun_2, Sun_3=sun_3, Sun_4=sun_4, P_1=pressure_1, P_2=pressure_2, P_3=pressure_3, P_4=pressure_4 )
    p_corr=@df data_avg corrplot([:T_1 :Sun_1 :P_1 :T_2])     
end
begin
    # Correlation Matrix on all the predictors
    using CategoricalArrays
    #Correlation between the predictors
    corMatrix_Predictors= broadcast(abs,cor(Matrix(train_std)))
    #Correlation between each predictors and the responses
    corMatrix_Responses=broadcast(abs,(cor(Matrix(train_std), levelcode.(train_cleaned.precipitation_nextday)[:,1])))
    names_station=names(train_cleaned)
    pop!(names_station);replace!(corMatrix_Responses, NaN=>0)
    corMatrix_Responses=hcat(corMatrix_Responses,names_station)  
    corMatrix_Responses=corMatrix_Responses[sortperm(corMatrix_Responses[:,1],rev=true),:]
end
begin
    #heatmap with the correlation matrix between predictors (50 first predicors)
    heatmap(1:50,1:50, (corMatrix_Predictors[1:50,1:50]),c=cgrad([:blue, :white,:red, :yellow]), xlabel="Predictors", ylabel="Predictors",title="Correlation between the classifier", figsize=(1000,1000))
end
begin
    #Bar plot with the pearson correlation coefficient (PCC) of the 20 predictors with the highest correlation. The corroletion is computed between the responses and a predictors
    bar(corMatrix_Responses[1:20,1],orientation=:h,title="10 best predictors (PCC)",yticks=(1:20, corMatrix_Responses[1:20,2]), yflip=true, label="PCC")
end

### Perform a PCA
begin
    #Generates the PCA
    using StatsBase,MLJMultivariateStatsInterface
    pca_train= (Matrix(train_std))
    replace!(pca_train, Inf=>NaN)
    replace!(pca_train, NaN=>0)
    pca_label=train_cleaned.precipitation_nextday
    mach_pca=fit!(machine(PCA(pratio=0.90),pca_train))
    pca_train=MLJ.transform(mach_pca,pca_train)
    print("Number of PCA for a 90% of variances explained: ",length(pca_train))
    plot_data=DataFrame(PC1=(pca_train.x1),
                            PC2=(pca_train.x2), 
                            PC3=pca_train.x3,
                            PC4=pca_train.x4,
                            lab=pca_label)
end
begin
    #Plot PCA1 - PCA2
    scatter(plot_data[plot_data.lab .==true,"PC1"], plot_data[plot_data.lab .==true,"PC2"], label="Rainning next day",xlabel="PCA1", ylabel="PCA2", color="red")
    scatter!(plot_data[plot_data.lab .==false,"PC1"], plot_data[plot_data.lab .==false,"PC2"], label="Not rainning next day",xlabel="PCA1", ylabel="PCA2", color="blue")
end
begin
    #Plot PCA1 - PCA3
    scatter(plot_data[plot_data.lab .==true,"PC1"], plot_data[plot_data.lab .==true,"PC4"], label="Rainning next day",xlabel="PCA1", ylabel="PCA3", color="red")
    scatter!(plot_data[plot_data.lab .==false,"PC1"], plot_data[plot_data.lab .==false,"PC4"], label="Not rainning next day",xlabel="PCA1", ylabel="PCA3", color="blue")
end

### Data augmentation
# Each time that two consecutive data points are labeled true or false, we generate a new point with the means value of the two consecutive points
begin
    train_1=CSV.read(joinpath(@__DIR__, "..", "data", "trainingdata.csv"), DataFrame)
    coerce!(train_1,:precipitation_nextday => Multiclass)
    train_filled_1= MLJ.transform(fit!(machine(FillImputer(), train_1)), train_1)
    #Generates the augmented data
    new_data=zeros(Float64,1,529)
    for i=1:3175
        if(train_filled_1.precipitation_nextday[i]==1 && train_filled_1.precipitation_nextday[i+1]==1)
            new_point=zeros(Float64,529)
            for j=1:528
            new_point[j]= (train_filled_1[i+1, j]-train_filled_1[i, j])/2
            end
            new_point[529]=true
            new_data=vcat(new_data, transpose(new_point))
        end

        if(train_filled_1.precipitation_nextday[i]==0 && train_filled_1.precipitation_nextday[i+1]==0)
            new_point=zeros(Float64,529)
            for j=1:528
            new_point[j]= (train_filled_1[i+1, j]-train_filled_1[i, j])/2
            end
            new_point[529]=false
            new_data=vcat(new_data, transpose(new_point))
        end
    end
    
    new_data[1,:]=new_data[size(new_data)[1],:]
    new_data = new_data[setdiff(1:end, 1), :]
    new_data=DataFrame( new_data, :auto)
    rename!(new_data, names(train_filled_1))
    print("Size of the created data :", size(new_data))
    #Export the augmented data
    augmented_data=vcat(train_filled_1, new_data)
    CSV.write(joinpath(@__DIR__, "..", "data", "augmenteddata.csv"), augmented_data)
end
