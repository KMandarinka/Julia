using Pipe
using StatsBase
using Distributions
using MLDatasets: Iris
# using Plots
using StatsPlots

function features_table((features_mtx, labels);)
  features_rows = map(collect, eachrow(features_mtx))

  labeled_features = map(features_rows) do row
    zip(labels, row) |> collect
  end

  (
    zip(
      [:sepal_length,
       :sepal_width,
       :petal_length,
       :petal_width],
      labeled_features
    ) |> Dict,
    labels
  )
end

function deep_label((table, labels);)
  final_table = Dict{Symbol, Dict{String, Vector{Float64}}}()

  foreach(table) do (param, features)
    final_table[param] = Dict()

    map(unique(labels)) do label
      partial_features = filter(features) do (inner_label, _)
        inner_label == label
      end
      final_table[param][label] = map(p -> p[2], partial_features)
    end
  end

  final_table
end

parse_features = deep_label ∘ features_table

full_table = (Iris.features(), Iris.labels()) |> parse_features


ivc_pw = full_table[:petal_width]["Iris-versicolor"]







versicolor_petal_w_nd = fit(Normal, ivc_pw)



name = "katyoosha"
μ = mean(versicolor_petal_w_nd)
σ = std(versicolor_petal_w_nd)

println("Математическое ожидание petal width Iris-versicolor ", "$μ")
println("Стандартное отклонение petal width Iris-versicolor ", "$σ")

plot(versicolor_petal_w_nd, label="$name,\n μ = $μ,\n σ = $σ") |> display

