#MammaPrint parte I 
#Mario Brea
setwd('C:/Users/224D2000/Documents/Act_ML')
#Instalación de paquetes necesarios 
if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("readr", "caret", "lattice")

install.packages(c("readr", "caret", "lattice"), dependencies = TRUE)
#Mejor esto porqeu son paquetes de CRAN
library(readr)
library(caret)
library(lattice)

#We create both metadata and expresion matrix
mat<- read.delim("matriz4ML_MammaPrint.tsv",
                 sep = "\t", header = TRUE, stringsAsFactors = FALSE)
clinical<- read.delim("clinical_info_TCGA-BRCA_MammaPrintInfo.tsv",
                      sep = "\t", header = TRUE, stringsAsFactors = FALSE)

identical(rownames(clinical), rownames(mat)) #comprobamos si son mismo orden de muestras en clinical y mat

#We create risk category: $MP_risk, with labels 0 and 1, indicating low or high risk.
riesgo<-factor(clinical$MP_risk, levels = c(0,1), labels = c("Bajo", "Alto"))
table(riesgo)
str(clinical$MP_risk)

#We take 5 significant genes randomly out of a 20 gene list 
#of one previous differential expression analysis made
listgenessig<-c("C11orf86" ,"ODAM", "C1QL2" ,"LCTL", "FMO6P", "LEMD1", "CLDN16", "RLBP1" ,"LOC93432", "BPI", "SMR3B" ,"GFRA3" ,"MMP20", "OCA2", "CXorf49B", "KLHL34", "MPZ", "DMRTA2" ,"CLDN6", "RASGEF1C")
set.seed(123)
genes_azar<- sample(listgenessig, 5)
submat<-mat[,genes_azar]

# Scatter plots by pairs
featurePlot(x = submat, y = riesgo, plot = "pairs",
            auto.key = list(columns = 2, title = "Riesgo"),
            cex= 1)

# Boxplots
featurePlot(x = submat, y = riesgo, plot = "box",
            auto.key = list(columns = 2, title = "Riesgo"))

# Density plots
featurePlot(x = submat, y = riesgo, plot = "density",
            auto.key = list(columns = 2, title = "Riesgo"))

#Removal of low variance genes 
nzv<- nearZeroVar(mat)
length(nzv) #cuantos quitamos        
mat_nzv<-mat[,-nzv] #subset new`matrix`

#Removal of high correlation genes.
cor_mat<-cor(mat_nzv)
high_cor_mat<-findCorrelation(cor_mat, cutoff=0.75)
length(high_cor_mat)
mat_filtered<-mat_nzv[,-high_cor_mat] #subset de la matriz son varianza, ahora quitando cor

saveRDS(mat_filtered, "X_filtrada.rds")  # predictores, variables independdientes
saveRDS(riesgo,   "y_riesgo.rds")    # variable objetivo


# Environment save.
save.image(file = "myenv_MammaPrint_PartI.RData")

