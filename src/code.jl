using CSV, DataFrames, Random, Images, FileIO, Flux, Statistics, ProgressMeter, MLUtils

# Random seed for reproducibility
Random.seed!(42)

# Load and filter the dataset
styles = CSV.read("Fashion_mnist/styles.csv", DataFrame)

# Keep selected subcategories
selected_subcats = ["Topwear", "Bottomwear", "Innerwear", 
                   "Shoes", "Sandal", "Flip Flops",
                   "Bags", "Socks", "Scarves"]

filtered = styles[styles.subCategory .∈ [selected_subcats], :]

# Create sampled dataset with exact counts
subcat_counts = Dict(
    "Topwear" => 230,
    "Bottomwear" => 40,
    "Innerwear" => 30,
    "Shoes" => 70,
    "Sandal" => 10,
    "Flip Flops" => 10,
    "Bags" => 5,
    "Socks" => 3,
    "Scarves" => 2
)

sampled = DataFrame()
for (subcat, n) in subcat_counts
    sub_df = filtered[filtered.subCategory .== subcat, :]
    if nrow(sub_df) < n
        @warn "Only $(nrow(sub_df)) available for $subcat (requested $n)"
        append!(sampled, sub_df)
    else
        append!(sampled, sub_df[randperm(nrow(sub_df))[1:n], :])
    end
end

# Verify we have exactly 400 items
if nrow(sampled) > 400
    sampled = sampled[1:400, :]
elseif nrow(sampled) < 400
    needed = 400 - nrow(sampled)
    extra = filtered[filtered.subCategory .== "Topwear", :]
    extra = extra[randperm(nrow(extra))[1:needed], :]
    sampled = vcat(sampled, extra)
end

# Save the final dataset
CSV.write("fashion_400_top_categories.csv", sampled)

# Show final distribution
final_counts = combine(groupby(sampled, [:masterCategory, :subCategory]), nrow => :count)
sort!(final_counts, :count, rev=true)
println("Final distribution:")
println(final_counts)

# 2. IMAGE LOADING & PREPROCESSING (FIXED)
function load_images(df, image_dir)
    # Initialize arrays with proper dimensions
    num_images = nrow(df)
    X = Array{Float32}(undef, 64, 64, 3, num_images) 
    y = Int[]
    
    # Create label mapping
    categories = unique(df.subCategory)
    label_dict = Dict(cat => i for (i, cat) in enumerate(categories))
    
    for (i, row) in enumerate(eachrow(df))
        img_path = joinpath(image_dir, "$(row.id).jpg")
        try
            # Load and preprocess image
            img = load(img_path)
            img = imresize(img, (64, 64))
            
            # Convert to array and handle grayscale/RGB
            if ndims(img) == 2  # Grayscale
                X[:,:,1,i] .= Float32.(img) ./ 255f0
                X[:,:,2,i] .= Float32.(img) ./ 255f0
                X[:,:,3,i] .= Float32.(img) ./ 255f0
            else  
                X[:,:,:,i] .= permutedims(Float32.(channelview(img)), (2,3,1)) ./ 255f0
            end
            
            push!(y, label_dict[row.subCategory])
        catch e
            @error "Failed to process $(row.id): $e"
            X[:,:,:,i] .= 0f0  
            push!(y, 1) 
        end
    end
    
    # Convert to channel-first format (3,64,64,N)
    X = permutedims(X, (3, 2, 1, 4))
    y = Flux.onehotbatch(y, 1:length(label_dict))
    
    return (X, y), label_dict
end

# Load and preprocess images
image_dir = "Fashion_mnist/images" 
(data, labels), label_dict = load_images(sampled, image_dir)

# Verify dimensions
println("\nData dimensions: ", size(data)) 
println("Label dimensions: ", size(labels))

# 3. MCNN9 ARCHITECTURE (UNCHANGED)
function MCNN9(num_classes=9)
    Chain(
        Conv((3, 3), 3=>256, pad=(1, 1), relu),
        Conv((3, 3), 256=>256, pad=(1, 1), relu),
        Conv((3, 3), 256=>192, pad=(1, 1), relu),
        MaxPool((2, 2)),
        Conv((3, 3), 192=>256, pad=(1, 1), relu),
        Conv((3, 3), 256=>32, pad=(1, 1), relu),
        Conv((3, 3), 32=>192, pad=(1, 1), relu),
        MaxPool((2, 2)),
        Conv((3, 3), 192=>192, pad=(1, 1), relu),
        Conv((3, 3), 192=>128, pad=(1, 1), relu),
        Conv((3, 3), 128=>32, pad=(1, 1), relu),
        MaxPool((2, 2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(32*8*8, 64, relu),
        Dense(64, num_classes),
        softmax
    )
end

# 4. FIXED TRAINING LOOP WITH 10-FOLD CV
function run_10fold_cv()
    X, y = load_and_preprocess_data()
    println("Shape of X before permutedims: ", size(X))
    #X = permutedims(X, (2, 3, 1, 4))  
    println("Shape of X after permutedims: ", size(X))

    k = 10
    N = size(X, 4)
    indices = shuffle(1:N)
    fold_size = div(N, k)
    accuracies = []

    for fold in 1:k
        println("\n--- Fold $fold ---")

        test_idx = indices[(fold-1)*fold_size+1 : fold*fold_size]
        train_idx = setdiff(indices, val_idx)

        X_train, y_train = X[:, :, :, train_idx], y[:, train_idx]
        X_test,   y_test   = X[:, :, :, test_idx],   y[:, test_idx]

        println("X_train shape: ", size(X_train))
        println("y_train shape: ", size(y_train))
        println("X_test shape: ", size(X_test))
        println("y_test shape: ", size(y_test))

        model = MCNN9(size(y, 1))
        opt = Flux.setup(Flux.Adam(0.001), model)

        # Train
        for epoch in 1:5
            loader = Flux.DataLoader((X_train, y_train), batchsize=32, shuffle=true)
            for (x, y) in loader
                println("Batch X: ", size(x))
                println("Batch y: ", size(y))
                gs = Flux.gradient(model) do m
                    Flux.crossentropy(m(x), y)
                end
                Flux.update!(opt, model, gs)
            end
        end

        # Validate
        preds = model(X_test)
        acc = mean(Flux.onecold(preds) .== Flux.onecold(y_test))
        println("Accuracy: $(round(acc * 100, digits=2))%")
        push!(accuracies, acc)
    end

    println("\n--- 10-Fold Results ---")
    println("Fold Accuracies: ", [round(a*100, digits=2) for a in accuracies])
    println("Average Accuracy: $(round(mean(accuracies)*100, digits=2))%")
end

# MAIN
X, y = load_and_preprocess_data()
println("Loaded X shape: ", size(X))  # (64,64,3,N)
println("Loaded y shape: ", size(y))  # (classes,N)

# X = permutedims(X, (2, 3, 1, 4))
model = MCNN9(size(y,1))
opt = Flux.setup(Flux.Adam(0.001), model)
loader = Flux.DataLoader((X,y), batchsize=32)
for (x, y) in loader
    println("Main Batch X: ", size(x))
    println("Main Batch y: ", size(y))
    gs = Flux.gradient(model) do m
        Flux.crossentropy(m(x), y)
    end
Flux.update!(opt, model, gs)
end

# Run CV
run_10fold_cv()

using Plots

plot(1:10, accuracies .* 100,
     label = "Accuracy",
     xlabel = "Fold",
     ylabel = "Accuracy (%)",
     title = "10-Fold Cross Validation Accuracy",
     lw = 2,
     marker = :circle)