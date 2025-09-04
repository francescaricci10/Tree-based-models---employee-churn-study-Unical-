# --- MASTER DI IN ARTIFICIAL INTELLIGENCE AND DATA SCIENCE 2025 - UNICAL --- #
# ---------------------- TREE BASED MODELS - A.GOTTARD ---------------------- #

      # --------- RANDOM SURVIVAL FOREST - EMPLOYEE CHURN ---------- #
      # ------ MARCO LONGO - FRANCESCA RICCI - MARIA ROTELLA ------- #




##### DATASET LOADING AND PREPROCESSING

library(readr)
library(dplyr)

temp.df <- read.csv("dataset\\hr.csv")

# Rimozione di PROMOTION_LAST_5YEARS: significatività estremamente bassa
hr.df <- temp.df %>% select(-promotion_last_5years)

rm(temp.df)





##### SURVIVAL ANALYSIS PLOTS --------------------------------------------------

library(survival)
library(survminer)

# Censoring and events
set.seed(123)

df.sample <- hr.df %>% sample_n(50)
df.sample$Index <- 1:nrow(df.sample)

ggplot(df.sample, aes(x = time_spend_company, y = Index)) +
  geom_segment(aes(x = 0, xend = time_spend_company, y = Index, yend = Index), color = "black") +
  geom_point(aes(x = time_spend_company, color = factor(left)), size = 3) +  
  scale_color_manual(values = c("blue3", "red2"), labels = c("Censoring", "Event")) +
  scale_x_continuous(
    breaks = seq(min(df.sample$time_spend_company)-1, max(df.sample$time_spend_company), by = 6)) + 
  labs(x = "months", y = "", title = "Employee churn (random sample of 50 employees)") +
  theme_minimal() +
  theme(legend.position = "bottom", legend.title = element_blank())

rm(df.sample)

# Survival Curve
surv.object <- Surv(time = hr.df$time_spend_company, event = hr.df$left)
km.fit <- survfit(surv.object ~ 1, data = hr.df)
ggsurvplot(km.fit, 
           title = "Survival curve", xlab = "time spend in company (months)", ylab = "",
           conf.int = TRUE, risk.table = FALSE,
           ggtheme = theme_classic(),
           palette = "Dark2", legend.title = "",legend = "none",
           censor.shape = 124, censor.size = 3, surv.median.line = "none"
)





##### TRAIN/TEST SETS - 70/30 --------------------------------------------------

set.seed(123)
train_indices <- sample(1:nrow(hr.df), size = floor(0.7 * nrow(hr.df))) # 70%
df.train <- hr.df[train_indices, ] 
df.test <- hr.df[-train_indices, ] 






##### CART ---------------------------------------------------------------------

# Classification with Rpart
library(rpart)
library(rpart.plot)

decision.tree <- rpart(left ~ ., 
                       data = df.train, 
                       method = "class")
rpart.plot(decision.tree,
           type = 3, extra = 104, nn = TRUE, 
           box.palette = "RdBu", shadow.col = "gray")
summary(decision.tree)

# Survival with Rpart
library(survival)

survival.tree <- rpart(
  Surv(event=df.train$left, time=df.train$time_spend_company) ~ ., 
  data = df.train, 
  method = "exp")

rpart.plot(survival.tree,
           type = 3, extra = "auto", nn = TRUE,
           box.palette = "RdBu", shadow.col = "gray")

summary(survival.tree)

# Prediction
pred.survival <- predict(survival.tree, newdata = df.test, type = "vector")

# C index
c.index <- concordance(Surv(df.test$time_spend_company, df.test$left)~pred.survival)
print(c.index$concordance)

# ROC and AUC
library(pROC)
roc.curve <- roc(df.test$left, pred.survival)

plot(roc.curve, 
     col = "blue", lwd = 2,             
     main = "ROC Curve", 
     xlab = "False Positive Rate", ylab = "True Positive Rate")  
#abline(a = 0, b = 1, col = "red", lwd = 2, lty = 2) 
auc_value <- auc(roc.curve)
legend("bottomright", legend = paste("AUC =", round(auc_value, 4)), 
       col = "blue", lwd = 2, bty = "n") 








##### RANDOM SURVIVAL FOREST ---------------------------------------------------

library(randomForestSRC)

df.train[] <- lapply(df.train, function(x) {
  if (is.character(x)) as.factor(x) else x
})


# Parameter tuning 
tune.result <- tune.rfsrc(Surv(time = time_spend_company, event = left) ~ ., 
                          mtryStart = ncol(df.train) / 2,
                          data = df.train, nodesizeTry = 15, ntreeTry = 500)
tune.result

# RANDOM SURVIVAL FOREST
B <- 1000
set.seed(1)

##### Splitting rule: LOGRANK
obj.logrank <- rfsrc(Surv(time_spend_company, left) ~ ., 
                     df.train, ntree = B, 
                     nodesize = 15, mtry = 5, importance = TRUE, 
                     splitrule = "logrank")
print(obj.logrank)
plot(obj.logrank)

# first tree
single.tree1 <- get.tree(obj.logrank, 1)
plot(single.tree1)

# forest
plot.survival(obj.logrank, cens.model = "rfsrc")


##### Splitting rule: LOGRANK-SCORE
obj.logrankscore <- rfsrc(Surv(time_spend_company, left) ~ ., 
                          df.train, ntree = B, 
                          nodesize = 15, mtry = 5, importance = TRUE, 
                          splitrule = "logrankscore")
print(obj.logrankscore)
plot(obj.logrankscore)

# first tree
single.tree2 <- get.tree(obj.logrankscore, 1)
plot(single.tree2)

# forest
plot.survival(obj.logrankscore, cens.model = "rfsrc")



# ------ Valutazione della bontà dei modelli ------ #

# The Concordance Index (C-Index) measures the model's ability to correctly rank 
# pairs of observations based on the predicted survival time.

# The Out-Of-Bag Error (err.rate) returned by rfsrc represents (1 - C-Index)

# OOB c index
cindex.oob <- 1 - obj.logrank$err.rate[length(obj.logrank$err.rate)]
cindex.oob

cindex.score.oob <- 1 - obj.logrankscore$err.rate[length(obj.logrankscore$err.rate)]
cindex.score.oob


#BiocManager::install("survcomp")
library(survcomp)
set.seed(1)
cindex.logrank <- concordance.index(
  x = predict(obj.logrank, newdata=df.test)$predicted,
  surv.time = df.test$time_spend_company,
  surv.event = df.test$left
)
print(cindex.logrank$c.index) # 0.9138987

set.seed(1)
cindex.logrank.score <- concordance.index(
  x = predict(obj.logrankscore, newdata=df.test)$predicted,
  surv.time = df.test$time_spend_company,
  surv.event = df.test$left
)
print(cindex.logrank.score$c.index) # 0.8860353


results <- data.frame(
  Model = c("Logrank", "Logrank Score"),
  Splitrule = c("logrank", "logrankscore"),
  OOB_CIndex = round(1 - c(
    obj.logrank$err.rate[length(obj.logrank$err.rate)],
    obj.logrankscore$err.rate[length(obj.logrankscore$err.rate)]
  ), 4),
  C_Index = round(c(
    cindex.logrank$c.index,
    cindex.logrank.score$c.index
  ), 4)
)

print(results)

# Visti i risultati procediamo solo con LOG-RANK

oo.logrank <- subsample(obj.logrank, verbose = FALSE)

vimpCI.logrank <- extract.subsample(oo.logrank)$var.jk.sel.Z
vimpCI.logrank

par(mar = c(5, 10, 1, 1))
plot(oo.logrank, xlim = c(0, 1))


plot.variable(obj.logrank, xvar.names = "satisfaction_level", partial = TRUE)
plot.variable(obj.logrank, xvar.names = "average_monthly_hours", partial = TRUE)
plot.variable(obj.logrank, xvar.names = "last_evaluation", partial = TRUE)
plot.variable(obj.logrank, xvar.names = "number_project", partial = TRUE)


##### BRIER SCORE

# Formula: Brier Score = (1/N) * sum((y_i - p_i)^2)
# y_i evento osservato, p_i probabilità stimata

bs.km.logrank <- get.brier.survival(obj.logrank, cens.mode = "km")$brier.score
bs.rsf.logrank <- get.brier.survival(obj.logrank, cens.mode = "rfsrc")$brier.score

graphics.off()

plot(bs.km.logrank, type = "s", col = 2, main = "Brier Scores", 
     cex.main = 1.2, cex.axis = 1, xlab = "Time", ylab = "Brier Score")
lines(bs.rsf.logrank, type = "s", col = 4)
legend("bottomright", legend = c("cens.model = km", "cens.model = rfsrc"), fill = c(2, 4))



##### CRPS SCORE

# Formula: CRPS = integral((F_hat(x) - F(x))^2) dx
# F_hat è la distribuzione cumulativa prevista e F è quella osservata

trapz.logrank <- randomForestSRC:::trapz
time.logrank <- obj.logrank$time.interest

crps.km.logrank <- sapply(1:length(time.logrank), function(j) {
  trapz.logrank(time.logrank[1:j], bs.km.logrank[1:j, 2] / diff(range(time.logrank[1:j])))
})
crps.rsf.logrank <- sapply(1:length(time.logrank), function(j) {
  trapz.logrank(time.logrank[1:j], bs.rsf.logrank[1:j, 2] / diff(range(time.logrank[1:j])))
})

graphics.off()

plot(time.logrank, crps.km.logrank, ylab = "CRPS", type = "s", col = 2, 
     main = "CRPS", cex.main = 1.2, cex.axis = 1, xlab = "Time")
lines(time.logrank, crps.rsf.logrank, type = "s", col = 4)
legend("bottomright", legend = c("cens.model = km", "cens.model = rfsrc"), fill = c(2, 4))




##### PREDICTION

pred.logrank <- predict(obj.logrank, newdata = df.test)

print(df.test[1,])
plot(pred.logrank$time.interest, pred.logrank$survival[1, ],
     type = "l", col = "blue", lwd = 2,
     xlab = "time", ylab = "Survival Probability",
     main = "prediction")

print(df.test[2,])
lines(pred.logrank$time.interest, pred.logrank$survival[2, ],
      type = "l", col = "red", lwd = 2)

print(df.test[3,])
lines(pred.logrank$time.interest, pred.logrank$survival[3, ],
      type = "l", col = "green", lwd = 2)

legend("topright", legend = c("emp1", "emp2", "emp3"),
       col = c("blue", "red", "green"), lwd = 2)


selected_indices_satisfaction_min <- which(df.test$satisfaction_level == min(df.test$satisfaction_level))
selected_indices_satisfaction_max <- which(df.test$satisfaction_level == max(df.test$satisfaction_level))

print(df.test[selected_indices_satisfaction_min[1], ])
print(df.test[selected_indices_satisfaction_max[2], ])

plot(pred.logrank$time.interest, pred.logrank$survival[selected_indices_satisfaction_min[1], ],
     type = "l", col = "blue", lwd = 2,
     xlab = "Time", ylab = "Survival Probability",
     main = "Satisfaction: Minimum vs Maximum Comparison")
lines(pred.logrank$time.interest, pred.logrank$survival[selected_indices_satisfaction_max[2], ],
      type = "l", col = "red", lwd = 2)
legend("bottomleft", legend = c("minimum satisfaction", "maximum satisfaction"),
       col = c("blue", "red"), lwd = 2)


selected_indices_salary_low <- which(df.test$salary == "low")
selected_indices_salary_high <- which(df.test$salary == "high")

print(df.test[selected_indices_salary_low[1], ])
print(df.test[selected_indices_salary_high[2], ])

plot(pred.logrank$time.interest, pred.logrank$survival[selected_indices_salary_low[1], ],
     type = "l", col = "green", lwd = 2,
     xlab = "Time", ylab = "Survival Probability",
     main = "Salary: Low vs High Comparison")
lines(pred.logrank$time.interest, pred.logrank$survival[selected_indices_salary_high[2], ],
      type = "l", col = "orange", lwd = 2)
legend("topright", legend = c("low salary", "high salary"),
       col = c("green", "orange"), lwd = 2)


selected_indices_projects_min <- which(df.test$number_project == min(df.test$number_project))
selected_indices_projects_max <- which(df.test$number_project == max(df.test$number_project))

print(df.test[selected_indices_projects_min[1], ])
print(df.test[selected_indices_projects_max[1], ])

plot(pred.logrank$time.interest, pred.logrank$survival[selected_indices_projects_min[1], ],
     type = "l", col = "purple", lwd = 2,
     xlab = "Time", ylab = "Survival Probability",
     main = "Projects: Minimum vs Maximum Number Comparison")
lines(pred.logrank$time.interest, pred.logrank$survival[selected_indices_projects_max[1], ],
      type = "l", col = "cyan", lwd = 2)
legend("topright", legend = c("min num projects", "max num projects"),
       col = c("purple", "cyan"), lwd = 2)






#######################################################################
# ------ RSF con 500 alberi e senza la covariata WORK_ACCIDENT ------ #

hr.df2 <- hr.df %>% select(-work_accident)

set.seed(1)
train.indices <- sample(1:nrow(hr.df2), size = floor(0.7 * nrow(hr.df2))) # 70%
df.train2 <- hr.df2[train.indices, ] 
df.test2 <- hr.df2[-train.indices, ] 


df.train2[] <- lapply(df.train2, function(x) {
  if (is.character(x)) as.factor(x) else x
})


# RANDOM SURVIVAL FOREST
B <- 500
set.seed(1)

##### Splitting rule: LOGRANK
obj.logrank2 <- rfsrc(Surv(time_spend_company, left) ~ ., 
                     df.train2, ntree = B, 
                     nodesize = 15, mtry = 5, importance = TRUE, 
                     splitrule = "logrank")
print(obj.logrank2)
plot(obj.logrank2)

# first tree
single.tree3 <- get.tree(obj.logrank2, 1)
plot(single.tree3)

# forest
plot.survival(obj.logrank2, cens.model = "rfsrc")


### C INDEX

cindex.logrank2 <- concordance.index(
  x = predict(obj.logrank2, newdata=df.test)$predicted,
  surv.time = df.test$time_spend_company,
  surv.event = df.test$left
)
print(cindex.logrank2$c.index) # 0.9264614


results2 <- data.frame(
  Model = c("Logrank_550", "Logrank_1000"),
  Splitrule = c("logrank", "logrank"),
  OOB_CIndex = round(1 - c(
    obj.logrank2$err.rate[length(obj.logrank2$err.rate)],
    obj.logrank$err.rate[length(obj.logrank$err.rate)]
  ), 4),
  C_Index = round(c(
    cindex.logrank2$c.index,
    cindex.logrank$c.index
  ), 4)
)

print(results2)


### V IMP

oo.logrank2 <- subsample(obj.logrank2, verbose = FALSE)

vimpCI.logrank2 <- extract.subsample(oo.logrank2)$var.jk.sel.Z
vimpCI.logrank2

par(mar = c(5, 10, 1, 1))
plot(oo.logrank, xlim = c(0, 1))


plot.variable(obj.logrank, xvar.names = "satisfaction_level", partial = TRUE)
plot.variable(obj.logrank, xvar.names = "average_monthly_hours", partial = TRUE)
plot.variable(obj.logrank, xvar.names = "last_evaluation", partial = TRUE)
plot.variable(obj.logrank, xvar.names = "number_project", partial = TRUE)



##### BRIER SCORE

# Formula: Brier Score = (1/N) * sum((y_i - p_i)^2)
# y_i evento osservato, p_i probabilità stimata

bs.km.logrank2 <- get.brier.survival(obj.logrank2, cens.mode = "km")$brier.score
bs.rsf.logrank2 <- get.brier.survival(obj.logrank2, cens.mode = "rfsrc")$brier.score

graphics.off()

plot(bs.km.logrank2, type = "s", col = 2, main = "Brier Scores", 
     cex.main = 1.2, cex.axis = 1, xlab = "Time", ylab = "Brier Score")
lines(bs.rsf.logrank2, type = "s", col = 4)
legend("bottomright", legend = c("cens.model = km", "cens.model = rfsrc"), fill = c(2, 4))



##### CRPS SCORE

trapz.logrank <- randomForestSRC:::trapz
time.logrank2 <- obj.logrank2$time.interest

crps.km.logrank2 <- sapply(1:length(time.logrank2), function(j) {
  trapz.logrank(time.logrank2[1:j], bs.km.logrank2[1:j, 2] / diff(range(time.logrank2[1:j])))
})
crps.rsf.logrank2 <- sapply(1:length(time.logrank2), function(j) {
  trapz.logrank(time.logrank2[1:j], bs.rsf.logrank2[1:j, 2] / diff(range(time.logrank2[1:j])))
})

graphics.off()

plot(time.logrank2, crps.km.logrank2, ylab = "CRPS", type = "s", col = 2, 
     main = "CRPS", cex.main = 1.2, cex.axis = 1, xlab = "Time")
lines(time.logrank2, crps.rsf.logrank2, type = "s", col = 4)
legend("bottomright", legend = c("cens.model = km", "cens.model = rfsrc"), fill = c(2, 4))

