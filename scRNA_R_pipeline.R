
# This is an automated scRNA-seq data analysis pipeline.

# Packages ----
#install.packages("Seurat")
#install.packages("tidyverse")
#install.packages("reshape2")
#install.packages("mclust")
#install.packages('installr')
#install.packages("rtools")
#install.packages("dplyr")
#install.packages("ggplot2")
#install.packages("Matrix")
#install.packages("factoextra")
#install.packages("knitr")
#install.packages("GGally")
#install.packages("viridis")
#install.packages("hrbrthemes")
#install.packages("RColorBrewer")
#install.packages("dbscan")
#install.packages("fpc")
#install.packages("R.utils")


library(rlang)
library(htmltools)
library(Seurat)
library(dplyr)
library(ggplot2)
library(knitr)
library(utils)
library(reshape2)
library(mclust, quietly = TRUE)
library(Matrix)
library(installr)
library(factoextra)
library(knitr)
library(tidyr)
library(hrbrthemes)
library(GGally)
library(viridis)
library(RColorBrewer)
library(R.utils)
library(data.table)
library(R.utils)
library(data.table)
library(cluster)
library(fpc)
library(dbscan)
library(readxl)



# Load dataset ----

load_dataset <- function(zip_file, csv_file) {
  
  # This function loads the dataset from a csv file.
  
  data <- read.csv(unz(zip_file, csv_file), sep = ",")
  print(head(data))
  
  return(data)
}


#
get_dataset_name <- function(csv_file){
  
  # This function extracts the dataset name from the csv filename.
  dataset_name <- tools::file_path_sans_ext(csv_file)
  
  return(dataset_name)
  
}


# Data exploration ----

# 
melt_dataset <- function(data) {
  
  # This function melts a dataset data frame into long format.
  
  melted_data <- reshape2::melt(data, value.name = 'expression')
  melted_data <- melted_data %>%
    rename(genes = "variable")
  
  return(melted_data)
  
}

#
get_library_size <- function(melted_data){
  
  # This function returns the library size for all cells in the dataset.
  
  library_size <- melted_data %>%
    group_by(cells) %>%
    summarise(library_size = sum(expression))
  
  print(library_size)
  
  return(library_size)
  
}
  
#
plot_GeneExpression_heatmap <- function(melted_data) {
  
  # This function creates a gene expression heatmap plot
  
  heatmap_plot <- ggplot(melted_data, aes(x = genes, y = cells, fill = expression)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "blue") +  
    labs(x = "Genes", y = "Cells", title = paste("Gene Expression Heatmap (", dataset_name, ")")) +
    theme(axis.text = element_blank())
  
  return(heatmap_plot)
  
}



# Data preprocessing ----

# Creation of Seurat object from the expression matrix data

#
build_seurat_object <- function(data, project_name){
  
  # This function builds a Seurat object from a scRNA seq expression matrix dataset.
  # Arguments:
  #     - data (matrix): gene expression matrix
  #     - project_name (str): desired naming of the project
  
  # Remove cell ID column from the expression matrix and make it index
  rownames(data) <- as.character(data[, 1])
  data <- data[, -1]
  
  # Transpose data matrix to be in correct format to be fit in a Seurat object
  data_transpose <- t(data)
  
  # Initialize the Seurat object
  data_seurat <- CreateSeuratObject(counts = data_transpose, project = project_name)
  
  return(data_seurat)
  
}


# Quality Control ----

# Step to identify low quality cells from the downstream analysis.

#
find_mitochondrial_genes <- function(seurat_object){
  
  # This function finds the mitochondrial genes in a Seurat object,
  # based on the "MT" label they have.
  seurat_object[["percent_mito"]] <- PercentageFeatureSet(seurat_object, pattern = "^MT")
  
  return(seurat_object)
  
}


#
plot_Features_violin <- function(seurat_object){
  
  # This function creates a violin plot of the specified features.
  
  vln_plot <- VlnPlot(seurat_object, features = c("nCount_RNA", "nFeature_RNA", "percent_mito"), ncol = 3)
  
  return(vln_plot)
  
}


#
plot_Features_scatter <- function(seurat_object){
  
  # This function creates a scatter plot of the specified features
  
  scatter_plot <- FeatureScatter(seurat_object, feature1 = "nCount_RNA", feature2 = "nFeature_RNA") +
    geom_smooth(method = 'lm')
  
  return(scatter_plot)
  
}


# Filtering ----

# Step to remove low quality cells from the downstream analysis.

# 
filter_cells <- function(seurat_object, nFeature_RNA_min = 160, nFeature_RNA_max = 1000, percent_mito_max = 5){
  
  # This function performs filtering to remove low quality cells, according to 
  # the specified filtering criteria.
  
  filtered_seurat <- subset(seurat_object, 
                            subset = nFeature_RNA > nFeature_RNA_min & 
                              nFeature_RNA < nFeature_RNA_max & 
                              percent_mito < percent_mito_max)
  
  return(filtered_seurat)
  
}


# Data Normalization ----

# The function NormalizeData from the Seurat package is called for this purpose. 


# Regress Out ----

#
perform_regress_out <- function(seurat_object){
  
  # This function performs regression-based normalization using Seurat's RegressOut function.
  # It regresses out the specified variables from the gene expression matrix.
  
  # Create a vector of variables to regress out
  latent_vars <- c("nCount_RNA", "percent_mito")
  
  # Perform regression-based normalization
  seurat_object <- ScaleData(seurat_object, vars.to.regress = latent_vars)
  
  return(seurat_object)
}



# Highly variable gene identification ----

#
FindPlot_variable_genes <- function(seurat_object){
  
  # This function finds the variable genes of the Seurat object and 
  # creates a plot showing the 10 most variable genes.
  
  seurat_object <- FindVariableFeatures(seurat_object, selection.method = "vst", nfeatures = 2000)
  
  top10_genes <- head(VariableFeatures(seurat_object), 10)
  
  plot_variable_genes <- VariableFeaturePlot(seurat_object)
  plot_variable_genes <- LabelPoints(plot = plot_variable_genes, points = top10_genes, repel = TRUE) +
    ggtitle(paste("Variable Genes Plot")) #of", dataset_name))
  
  print(plot_variable_genes)
  
  return(seurat_object)
  
}


# Data Scaling ----

# 
scale_data <- function(seurat_object){
  
  # This function scales the Seurat object expression data.
  
  all_genes <- rownames(seurat_object)
  seurat_object <- ScaleData(seurat_object, features = all_genes)
  
  return(seurat_object)
  
}


# Dimensionality Reduction ----

# It is performed in three different ways: PCA, UMAP, tSNE.

#
choose_principal_components <- function(seurat_object){
  
  # This function discovers in a quantitave way the optimum number of PCs to be used
  # for dimensionality reduction. 
  #   Metric 1: the point where the PCs only contribute 5% of stdv 
  #             and the PCs cumulatively contribute 90% of the stdv.
  #   Metric 2: the point where the percent change in variation between the consecutive PCs is less than 0.1%.
  
  pct <- seurat_object[["pca"]]@stdev / sum(seurat_object[["pca"]]@stdev) * 100
  # calculate cumulative percents for each PC
  cum_pct <- cumsum(pct)
  
  metric1 <- which(cum_pct > 90 & pct < 5)[1]
  metric2 <- sort(which((pct[1:length(pct) - 1] - pct[2:length(pct)]) > 0.1), decreasing = T)[1] + 1
  
  return(list(metric1, metric2))
  
}


#
reduce_dimensions <- function(seurat_object, technique){
  
  # This function reduces the dimensionality of the dataset using either PCA, UMAP or tSNE.
  # The user specifies which dimensionality reduction technique will be applied.
  # Arguments:
        # -seurat_object
        # -technique (str): dimensionality reduction technique to be applied
                            # aceepts: pca, umap, tsne
  
  if (technique == "pca"){
    seurat_object <- RunPCA(seurat_object, features = VariableFeatures(object = seurat_object))
    #possible_dimensions <- choose_principal_components(seurat_object)
    #chosen_dimensions <- min(possible_dimensions[[1]], possible_dimensions[[2]])
    
    cells_num <- seurat_object@assays[["RNA"]]@counts@Dim[2]
    pc_heatmap <- DimHeatmap(seurat_object, dims = 1:9, cells = cells_num, balanced = TRUE)
    
    elbow_plot <- ElbowPlot(seurat_object) +
      labs(title = paste("Elbow Plot for", dataset_name))
    dimplot_unclustered <- DimPlot(seurat_object, reduction="pca")
    print(pc_heatmap)
    print(elbow_plot)
    print(dimplot_unclustered)
    
  } else if (technique == "umap"){
    seurat_object <- RunPCA(seurat_object, features = VariableFeatures(object = seurat_object))
    possible_dimensions <- choose_principal_components(seurat_object)
    chosen_dimensions <- max(possible_dimensions[[1]], possible_dimensions[[2]])
    seurat_object <- RunUMAP(seurat_object, dims = 1:chosen_dimensions)
    
    
  } else if (technique == "tsne") {
    seurat_object <- RunPCA(seurat_object, features = VariableFeatures(object = seurat_object))
    seurat_object <- RunTSNE(seurat_object)
    
    
  } else {
    cat("Dimensionality reduction technique not found!\n")
    cat("Function accepts: pca, umap, tsne.\n")
    cat("Check spelling!")
  }
  
  return(seurat_object)
  
}



# Clustering ----

# 
cluster_MeansCovariances <- function(gmm_model){
  
  # This function prints the means and covariance matrices of the clusters.
  
  cluster_means <- gmm_model$parameters$mean
  cluster_covariances <- gmm_model$parameters$variance$sigma
  cluster_covariances_df <- as.data.frame(cluster_covariances)
  
  cat("\n")
  cat("Cluster means:\n")
  print(cluster_means)
  cat("\n")
  cat("Cluster covariance matrices:\n")
  print(cluster_covariances)
  
  #return(list(cluster_mean, cluster_covariances))
  
}


#
GMM_clustering <- function(seurat_object, dim_reduction_technique){
  
  # This function clusters the cells using Gaussian Mixture Models. 
  # The optimal GMM model is selected with the BIC criterion.
  
  if (dim_reduction_technique == "pca"){
    #possible_dimensions <- choose_principal_components(seurat_object)
    #chosen_dimensions <- min(possible_dimensions[[1]], possible_dimensions[[2]])
    
    embeddings <- as.matrix(seurat_object@reductions$pca@cell.embeddings)
    
    # perform GMM clustering and choose best model according to the BIC criterion
    gmm_model <- Mclust(embeddings[ ,1:10], G = 1:20)
    cluster_ids <- gmm_model$classification
    
    # assign the cluster identities to the Seurat object
    Idents(seurat_object) <- as.character(cluster_ids)
    seurat_object$cluster_label <- Idents(seurat_object)
    
    print(summary(gmm_model))
    cluster_MeansCovariances(gmm_model)
    
    # Perform t-SNE projection
    seurat_object <- RunTSNE(seurat_object, dims = 1:10)
    DimPlot(seurat_object, group.by = "cluster_label")
    
    
  } else if (dim_reduction_technique == "umap"){
   
    embeddings <- as.matrix(seurat_object@reductions$umap@cell.embeddings)
    
    # perform GMM clustering and choose best model according to the BIC criterion
    number_of_umaps <- ncol(embeddings)
    gmm_model <- Mclust(embeddings[ ,1:number_of_umaps], G = 1:10)
    cluster_ids <- gmm_model$classification
    
    # assign the cluster identities to the Seurat object
    Idents(seurat_object) <- as.character(cluster_ids)
    seurat_object$cluster_label <- Idents(seurat_object)
    
    print(summary(gmm_model))
    cluster_MeansCovariances(gmm_model)
    
    
  } else if (dim_reduction_technique == "tsne"){
    
    seurat_object <- RunTSNE(seurat_object)
    embeddings <- as.matrix(seurat_object@reductions$tsne@cell.embeddings)
    
    # perform GMM clustering and choose best model according to the BIC criterion
    number_of_tsnes <- ncol(embeddings)
    gmm_model <- Mclust(embeddings[ ,1:number_of_tsnes], G = 1:20)
    cluster_ids <- gmm_model$classification
    
    # assign the cluster identities to the Seurat object
    Idents(seurat_object) <- as.character(cluster_ids)
    seurat_object$cluster_label <- Idents(seurat_object)
    
    print(summary(gmm_model))
    cluster_MeansCovariances(gmm_model)
    
  } else {
    cat("Dimensionality reduction technique has not been applied!\n")
    cat("Function accepts: pca, umap, tsne.\n")
    cat("Check spelling!")
  }
  
  return(list(seurat_object, gmm_model, embeddings))
  
} 


#
perform_dbscan_clustering <- function(seurat_object, dim_reduction_technique) {
  if (dim_reduction_technique == "pca") {
    # Run PCA
    seurat_object <- RunPCA(seurat_object, features = VariableFeatures(object = seurat_object))
    
    # Get PCA embeddings
    embeddings <- seurat_object@reductions$pca@cell.embeddings
    
    # Perform DBSCAN clustering
    dbscan_result <- dbscan(embeddings, eps = 50, MinPts = 50)
    
    # Assign cluster labels to Seurat object
    seurat_object$cluster_label <- as.character(dbscan_result$cluster)
    
    # Plot t-SNE visualization
    seurat_object <- RunTSNE(seurat_object, dims = 1:10)
    DimPlot(seurat_object, group.by = "cluster_label")
    
    return(seurat_object)
  } else {
    stop("Unsupported dimensionality reduction technique.")
  }
}


# Visualization ----

# 
plot_clustering <- function(dataset_name, seurat_object, gmm_model){
  
  # This function creates five plots that showcase the clustering results.
  
  dim_reduction_name <- names(data_seurat@reductions)[length(data_seurat@reductions)]
  num_clusters <- length(unique(gmm_model$classification))
  color_palette <- brewer.pal(num_clusters, "Paired")
  
  # 1. Basic representation of the cells clustering
  dimplot <- DimPlot(seurat_object, group.by = "cluster_label") +
    ggtitle(paste("Clustered Cells (", dataset_name, "-", dim_reduction_name, ")")) +
    scale_color_manual(values = color_palette)
  print(dimplot)
  
  # 2. BIC values across the PCs
  bic_lineplot <- fviz_mclust_bic(gmm_model) +
    ggtitle(paste("Model selection (", dataset_name, "-", dim_reduction_name, ")"))
  print(bic_lineplot)
  
  # 3. More detailed clustering representation showing the cell IDs as well
  clusters_plot <- fviz_cluster(gmm_model) +
    ggtitle(paste("Cluster Plot (", dataset_name, "-", dim_reduction_name, ")")) +
    scale_color_manual(values = color_palette)
  print(clusters_plot)
  
  # 4. Clustering uncertainty plot
  uncertainty_plot <- fviz_mclust(gmm_model, 'uncertainty') +
    ggtitle(paste("Cluster Plot (", dataset_name, "-", dim_reduction_name, ")")) +
    scale_color_manual(values = color_palette)
  print(uncertainty_plot)
  
  # 5. Cluster density plot
  density_plot <- plot(gmm_model, what = "density") 
  title(main = paste("Density Plot (", dataset_name, "-", dim_reduction_name, ")"), line = +1)
  
  
  return(list(dimplot, bic_lineplot, clusters_plot, uncertainty_plot, density_plot))
  
}


# 
extract_probability_results <- function(gmm_model, cell_embeddings){
  
  # This function extracts the clustering results (classification, embeddings,
  # posterior probabilities) to be used in further plotting.
  
  posterior_data <- data.frame(
    cell_id = rownames(cell_embeddings),
    cluster_label = gmm_model$classification,
    posterior_prob = gmm_model$z
  )
  
  posterior_data_long <- posterior_data %>%
    pivot_longer(cols = starts_with("posterior_prob"),
                 names_to = "cluster",
                 values_to = "posterior_prob")
  
  return(list(posterior_data, posterior_data_long))
  
}


#
calculate_joint_probabilities <- function(posterior_data_long, gmm_model){
  
  # This function calculated the joint posterior probability of each cell.
  
  weights <- gmm_model$parameters$pro
  
  joint_probs <- posterior_data_long %>%
    group_by(cell_id, cluster_label) %>%
    summarise(joint_prob = prod(posterior_prob * weights[cluster_label]))
  
  return(joint_probs)
  
}


#
plot_post_probabilities <- function(dataset_name, posterior_data, posterior_data_long, joint_probs){
  
  # This function creates one plots of the posterior probabilities of the cells 
  # and three plots for their joint probabilities.
  dim_reduction_name <- names(data_seurat@reductions)[length(data_seurat@reductions)]
  num_clusters <- length(unique(cell_joint_probabilities$cluster_label))
  color_palette <- brewer.pal(num_clusters, "Set1")
  
  # 1. Parallel coordinates chart with the posterior cell probabilities across all states (clusters)
  post_probs_plot <- ggparcoord(posterior_data, columns = 3:ncol(posterior_data), groupColumn = "cluster_label",
             scale = "uniminmax", alphaLines = 0.5, showPoints = TRUE) +
    scale_color_viridis(discrete=FALSE) +
    labs(x = "Posterior Probabilities", y = "Probability Value", color = "Cluster Label",
         title = paste("Cell Posterior Probabilities (", dataset_name, "-", dim_reduction_name, ")")) 
  
    # Adjust the plot aesthetics and theme
    theme_set(theme_minimal())
  
    
  # 2. Density of joint cell probabilities across all clusters.
  density_plot1 <- ggplot(joint_probs, aes(x = joint_prob)) +
    geom_density(fill = "blue", alpha = 0.5) +
    labs(x = "Cell Joint Probability", y = "Density", 
         title = paste("Cell Joint Probability Distribution (", dataset_name, "-", dim_reduction_name, ")")) +
    theme_minimal()
  
  
  # 3. Density of joint cell probabilities for each cluster.
  density_plot2 <- ggplot(joint_probs, aes(x = joint_prob, fill = factor(cluster_label))) +
    geom_density(alpha = 0.5) +
    labs(x = "Joint Probability", y = "Density", 
         title = paste("Cell Joint Probability Distribution by Cluster (", dataset_name, "-", dim_reduction_name, ")")) +
    theme_minimal() +
    facet_wrap(~ cluster_label, ncol = 1) +
    guides(fill = guide_legend(title = "Cluster Label")) +
    scale_fill_manual(values = color_palette)
  
  # 4. Scatter plot of the cell joint probabilities
  scatter_plot <- ggplot(data = joint_probs) + 
    geom_point(mapping = aes(x = cell_id, y = joint_prob, color = factor(cluster_label))) +
    theme(axis.text.x = element_blank()) +
    labs(title = paste("Cell Joint Probabilities (", dataset_name, "-", dim_reduction_name, ")"),
         x = "Cell ID",
         y = "Joint Probability",
         color = "Cluster Label") +
    scale_color_manual(values = color_palette)
  
  
  return(list(post_probs_plot, density_plot1, density_plot2, scatter_plot))
  
}













