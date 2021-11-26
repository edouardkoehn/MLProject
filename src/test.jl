using CSV, DataFrames, Plots

#To run the entire code option+enter
#Creation of block codes (to run them ctrl + enter)
begin
    data_test=CSV.read(joinpath(@__DIR__, "..", "data", "testdata.csv"), DataFrame)
end

begin
    plot(data_test[:,1], data_test[:,2], title= "Snowboard trace", label="Skier 1")
end

begin
    using MLJ, DataFrames, MLJMultivariateStatsInterface, OpenML, Plots,
          LinearAlgebra, Statistics, Random, MLJClusteringInterface, StatsPlots,
          Distributions
end

#Definition of the data generator
function data_generator(d; n = 20)
    cluster1 = rand(MvNormal([0; -d; fill(0, 48)], [3; fill(.5, 49)]), n)
    cluster2 = rand(MvNormal([0; 0; fill(0, 48)], [3; fill(.5, 49)]), n)
    cluster3 = rand(MvNormal([0; d; fill(0, 48)], [3; fill(.5, 49)]), n)
    (data = DataFrame(vcat(cluster1', cluster2', cluster3'), :auto),
     true_labels = [fill(1, n); fill(2, n); fill(3, n)])
end

#Create the data
begin
    data, label= data_generator(30)
end
#Perform PCA
begin
    Pca=MLJ.transform(fit!(machine(PCA(maxoutdim = 50), data)), data)
end
#Plot PCA
begin
    scatter(Pca.x1, Pca.x2, xlabel="PCA1", ylabel="PCA2", c=label, label="GT")
end

begin
#Define the KMeans K=3
    mach1 = fit!(machine(KMeans(k = 3), data))
    prediction = MLJ.predict(mach1, data)
end

begin 
    scatter!(Pca.x1 .+10, Pca.x2, xlabel="PCA1", ylabel="PCA2", c=convert(Vector{Int},prediction), label="Kmean K=3")
end

begin
    #Define the KMeans K=3
        mach1 = fit!(machine(KMeans(k = 4), data))
        prediction2 = MLJ.predict(mach1, data)
        #Out to save machine
        MLJ.save("trained_machines_test.jlso", mach1)
end

begin 
    scatter!(Pca.x1 .+20, Pca.x2, xlabel="PCA1", ylabel="PCA2", c=convert(Vector{Int},prediction2), label="Kmean K=4")
end

