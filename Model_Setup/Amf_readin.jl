#function for reading in relevant things like estimation sample, transition probabilities, etc
function Readin_ind(dir::String)
    #estimation sample
    temp=CSV.read(
    "$dir/mydata.csv",
    DataFrames.DataFrame;
    )
    ind = Array{Float64,2}(temp)
    ind
end

#function for reading in relevant things like estimation sample, transition probabilities, etc
function Readin_ind_20(dir::String)
    #estimation sample
    temp=CSV.read(
    "$dir/mydata.csv",
    DataFrames.DataFrame;
    )
    temp_20=filter(row -> row.t <1989, temp)
    ind_20= Array{Float64,2}(temp_20)
end

function Readin_loc(dir::String)
    temp=CSV.read(
    "$dir/locdata.csv",
    DataFrames.DataFrame;
    )
    loc= Array{Float64,2}(temp)
    loc
end
