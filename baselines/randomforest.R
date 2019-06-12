library("survival", logical.return = TRUE)
library("pec", logical.return = TRUE)
library("prodlim", logical.return = TRUE)
library("randomForestSRC")

# clinicalplussurvival = read.csv('~/Documents/Research/Projects/tcga-brca/data/bgam/clinical+days+survival.csv')
# clinicalplussurvivalplusproteingenes = read.csv('~/Documents/Research/Projects/tcga-brca/data/bgam/clinical+days+survival+proteinexp.csv')
clinicalplussurvivalplusreducedgenes = read.csv('~/Documents/Research/Projects/tcga-brca/data/bgam/clinical+days+survival+expreduced.csv')
# clinicalplussurvivalplusfullgenes = read.csv('~/Documents/Research/Projects/tcga-brca/data/bgam/status+event_time+clinical+exp_full.csv')

print("Loaded all data.")

data(pbc, package = "randomForestSRC")
pbc.na = clinicalplussurvivalplusreducedgenes # EDIT THIS FOR DATASET

surv.f <- as.formula(Surv(days, status) ~ .)
pec.f <- as.formula(Hist(days,status) ~ 1)

# cox.obj <- coxph(surv.f, data = pbc.na)

print("Finished learning Cox model.")

# rfsrc.obj <- rfsrc(surv.f, pbc.na, nsplit = 10, ntree = 150)

# print("Finished learning RSF model.")

set.seed(17743)

# prederror.pbc <- pec(list(rfsrc.obj), data = pbc.na, formula = pec.f, splitMethod = "bootcv", B = 50)
# prederror.pbc <- pec(list(cox.obj,rfsrc.obj), data = pbc.na, formula = pec.f, splitMethod = "bootcv", B = 50)
# print(prederror.pbc)
# plot(prederror.pbc)

rfsrc.obj <- rfsrc(surv.f, pbc.na, nsplit = 10)
print("Finished learning RSF model.")
cat("out-of-bag Cox Analysis ...", "\n")
# cox.err <- sapply(1:100, function(b) {
#	if (b%%10 == 0) cat("cox bootstrap:", b, "\n")
#	train <- sample(1:nrow(pbc.na), nrow(pbc.na), replace = TRUE)
#	cox.obj <- tryCatch({coxph(surv.f, pbc.na[train, ])}, error=function(ex){NULL})
#	if (is.list(cox.obj)) {randomForestSRC:::cindex(pbc.na$days[-train], pbc.na$status[-train], predict(cox.obj, pbc.na[-train, ]))} else NA
#	})

cat("\n\tOOB error rates\n\n")
cat("\tRSF : ", rfsrc.obj$err.rate[rfsrc.obj$ntree], "\n")
# cat("\tCox regression : ", mean(cox.err, na.rm = TRUE), "\n")
