# ---------------- INSTALLING LIBRARIES REQUIRED ---------------------------
install.packages("caret")
install.packages("rpart")
install.packages("xgboost")
install.packages("dplyr") 
install.packages("randomForest")
install.packages("rpart.plot")
 install.packages("pROC")
 install.packages("ROSE")
 
 
 # ---------------- LOADING LIBRARIES REQUIRED ------------------------------
 library(caret)
 library(rpart)
 library(dplyr)
 library(ROSE)
 library(randomForest)
 library(rpart.plot)
 library(xgboost)
 # --------------------------------------------------------------------------
 
 # --------------------- Reading the Data set --------------------------------
 credit_card = read.csv(file = 'D:/8TH SEMESTER WORKSHEETS/RLAB/creditcard.csv')
 
 # --------------------- Viewing the data set --------------------------------
 View(credit_card)
 
 # --------------------- Displaying the columns -----------------------------
 names(credit_card)
 
 # --------------------- Printing the structure of data set ------------------
 str(credit_card)
 
 # --------------------- Printing 5-6 obs of the data set --------------------
 head(credit_card)
 
 # - Converting the Class to factor as it has 0 (non-frauds) and 1 (frauds) --
 credit_card$Class = as.factor(credit_card$Class)
 
 # -------------- Summarizing the count of the Frauds and Non-Frauds --------
 summary(credit_card$Class)
 
 # --------------------- Checking for any NA values -------------------------
 sum(is.na(credit_card))
 
 # -------------- Separating the frauds and non-frauds into new dfs ---------
 credit_card.true = credit_card[credit_card$Class == 0,]
 credit_card.false = credit_card[credit_card$Class == 1,]
 
 # --------- Data Visualization on the basis of physically imp features -----
 ggplot()+geom_density(data = credit_card.true,aes(x = Time),color="blue",
                       fill="blue",alpha=0.12)+
   geom_density(data = credit_card.false,aes(x = Time),color="red",fill="red",
                alpha=0.12)
 
 ggplot()+geom_density(data = credit_card.true,aes(x = Amount),color="blue",
                       fill="blue",alpha=0.12)+
   geom_density(data = credit_card.false,aes(x = Amount),color="red",fill="red",
                alpha=0.12)
 
 # --------- PIE CHART for comparing no.of frauds and non-frauds ------------
 
 labels = c("NON_FRAUD","FRAUD")
 labels = paste(labels,round(prop.table(table(credit_card$Class))*100,2))
 labels = paste0(labels,"%")
 pie(table(credit_card$Class),labels,col = c("blue","red"),
     main = "Pie Chart of Credit Card Transactions")
 
 # ---------------------- DATA SPLITTING -------------------------------------
 rows = nrow(credit_card)
 cols = ncol(credit_card)
 
 set.seed(39)
 credit_card = credit_card[sample(rows),1:cols]
 ntr = as.integer(round(0.8*rows))
 
 credit_card.train = credit_card[1:ntr,1:cols] # for train
 credit_card.test = credit_card[(ntr+1):rows,-cols] # for test input
 credit_card.testc = credit_card[(ntr+1):rows,cols] # for test data CLass
 
 credit_card.testc = as.data.frame(credit_card.testc)
 colnames(credit_card.testc)[1] = c("Class")
 
 # ----------------------- LOGISTIC REGRESSION ------------------------------
 glm_fit <- glm(Class ~ ., data = credit_card.train, family = 'binomial')
 pred_glm <- predict(glm_fit,credit_card.test, type = 'response')
 
 credit_card.testc$Pred = 0L
 credit_card.testc$Pred[pred_glm>0.5] = 1L
 credit_card.testc$Pred = factor(credit_card.testc$Pred)
 
 confusionMatrix(credit_card.testc$Pred,credit_card.testc$Class)
 
 roc.curve(credit_card.testc$Class,credit_card.testc$Pred,plotit = TRUE,
           col="#D6604D",main = "ROC curve for Logistic Regression Algorithm",
           col.main="#B2182B")
 
 # -------------------- DECISION TREE ALGORITHM -----------------------------
 tree = rpart(Class ~ .,data = credit_card.train,method = "class")
 pred_tree = predict(tree,credit_card.test)
 
 credit_card.testc$Pred = 0L
 credit_card.testc$Pred[pred_tree[,2]>0.5] = 1L
 credit_card.testc$Pred = factor(credit_card.testc$Pred)
 
 confusionMatrix(credit_card.testc$Pred,credit_card.testc$Class)
 
 rpart.plot(tree,cex=0.66,extra = 0,type=5,box.palette = "BuRd")
 
 roc.curve(credit_card.testc$Class,credit_card.testc$Pred,plotit = TRUE,
           col="red",main = "ROC curve for Decision Tree Algorithm",
           col.main="darkred")
 
 # -------------------- RANDOM FOREST ALGORITHM -----------------------------
 samp = as.integer(0.49*ntr)
 rF = randomForest(Class ~ . ,data =credit_card.train,ntree = 39,
                   samplesize = samp,maxnodes=44)
 rF_pred = predict(rF,credit_card.test)
 credit_card.testc$Pred = rF_pred
 
 confusionMatrix(credit_card.testc$Pred,credit_card.testc$Class)
 
 roc.curve(credit_card.testc$Class,credit_card.testc$Pred,plotit = TRUE,
           col="green",main = "ROC curve for Random Forest Algorithm",
           col.main="darkgreen")
 
 # ----------------------- XGBOOST ALGORITHM --------------------------------
 labels <- credit_card.train$Class
 y <- recode(labels, '0' = 0, "1" = 1)
 xgb <- xgboost(data = data.matrix(credit_card.train[,-31]), 
                label = y,
                eta = 0.1,
                gamma = 0.1,
                max_depth = 10, 
                nrounds = 300, 
                objective = "binary:logistic",
                colsample_bytree = 0.6,
                verbose = 0,
                nthread = 7,
                set.seed(42)
 )
 xgb_pred <- predict(xgb, data.matrix(credit_card.test))
 
 credit_card.testc$Pred = 0L
 credit_card.testc$Pred[xgb_pred>0.5] = 1L
 credit_card.testc$Pred = factor(credit_card.testc$Pred)
 
 confusionMatrix(credit_card.testc$Pred,credit_card.testc$Class)
 
 roc.curve(credit_card.testc$Class,credit_card.testc$Pred,plotit = TRUE,
           col="blue",main = "ROC curve for XGBoost Algorithm",
           col.main="darkblue")
 
 # ------------------------- THE END ----------------------------------------
 # Hence, we can say that XGBOOST Algorithm was successful in predicting most
 # of the frauds (87/95) with an accuracy score of 99.96% and AUC of 0.910