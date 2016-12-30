

<html>


<head>

</head>


<body>
<div id="pageTitle">
<h1>Practical Machine Learning Course Final Project</h1>
<text style="font-size:20px">Howard Kaplan, 1.1.2017</text><br>
Coursera - John Hopkins University (MOOC)

</div>
<h2>Background</h2>
<p>
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These types of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 
</p>
<h2>Project Goal</h2>
<p>
The goal of your project is to predict the manner in which the 20 participants did the exercise (Unilateral Dumbbell Biceps Curl).
This is the "classe" variable in the training set. Using the training set I test various training models, access the outcomes, and then apply the final model(s) to the test data, to generate the prediction. <br>The out-of-sample error will be based on the total accuracy possible 1(100%) - the accuracy value produced by the predictive model.</p>
The classe or categories that the manner of the exercise fall under are as follows:<br>

<ul>
  <li>Class A: &nbsp;&nbsp;&nbsp;&nbsp;Exactly according to the specification</li>
  <li>Class B: &nbsp;&nbsp;&nbsp;&nbsp;Throwing the elbows to the front</li>
  <li>Class C: &nbsp;&nbsp;&nbsp;&nbsp;Lifting the dumbbell only halfway</li>
  <li>Class D: &nbsp;&nbsp;&nbsp;&nbsp;Lowering the dumbbell only halfway</li>
  <li>Class E: &nbsp;&nbsp;&nbsp;&nbsp;Throwing the hips to the front</li>
</ul>
<h2>R Code, Analysis and Results</h2>
First I need to download the data and read it into R
<pre class="code">
rm(list=ls())
###
#GET FILES
# Download data.
url_raw_training <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv?accessType=DOWNLOAD"
url_raw_testing <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv?accessType=DOWNLOAD"
#Read data CSVs
file_dest_training <- "pml-training.csv"
file_dest_testing <- "pml-testing.csv"

# Import the data to dataframe, treating empty values as NA.
df_training <- read.csv(file_dest_training, na.strings=c("NA","#DIV/0!",""), header=TRUE, sep=',')
#dim(df_training)
df_testing <- read.csv(file_dest_testing, na.strings=c("NA","#DIV/0!",""), header=TRUE, sep=',')
#dim(df_testing)
</pre>
Next I explore and clean the data.<br>Training dataset has 19622 rows and 160 variables (columns). Testing dataset has 20 rows and 160 variables (columns).<br> There was also a lot of "NA" values, and unnecessary variable/columns.
<pre class="code">
##Explore, Process and Clean Data
head(df_training)
colnames(df_training)

delRange = 1:7
#No need to use first 7 rows-(id,time,date,window, other useless data)
nTrain = df_training[,-delRange]; nTest = df_testing[,-delRange]
colnames(nTrain)

nTrain$classe <- as.factor(nTrain$classe); 
nTrainPrePross = nTrain
#get rid of NA
nTrainPrePross <- nTrainPrePross[, unlist(lapply(nTrainPrePross, function(x) !any(is.na(x))))]
</pre>
Once cleaned the training data (nTrainPrePross) is - 19622    53<br><br>
Next, I Load packages, Caret package and Rattle Package. And, Create/partition Train 60% and Test 40% sets from pml training data. The partioned data will be used for cross-validation. 
<pre class="code">
###############
##Load packages
library(caret); library(rattle)
#Create Training partitions or train and Test
inTrain <- createDataPartition(y=nTrainPrePross$classe, p=0.6, list=FALSE)
myTrain <- nTrainPrePross[inTrain, ]
myTest  <-  nTrainPrePross[-inTrain, ]
dim(myTrain);dim(myTest)
##Look at data
par(mfrow=c(1,2))
hist(as.numeric(myTrain$classe), main = "", labels = T, breaks = 6)
plot(myTrain$classe)
table(myTrain$classe)
</pre>
The partitions:
[1] 11776    53
[1] 7846   53
<br>
<img src="https://pszfdq.by3301.livefilestore.com/y3mFFhs6HQDIbGuqhD2lB-FGyQ6ulKN3lg1UK2IM6YBJTy62carPQUxcIYDdsz-xpi_mzO205eyfAyBl6gwpK3D_JXhwFebydetGgyO68y79xLffmaRVtdO6ZTTix0X5JGURAKTNTh15qlysdkJTeMY7YkxNmCTE6sVNSvy1TaRoKY?width=455&height=385&cropmode=none" alt="expPlot">
<br>   A &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     B &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    C &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;   D  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   E 
<br> 3348&nbsp;&nbsp;&nbsp;  2279&nbsp;&nbsp;&nbsp;  2054&nbsp;&nbsp;&nbsp;  1930&nbsp;&nbsp;&nbsp;  2165 
<br><br>Since the data needs to be categorized I decided to use a Decision Tree / Recursive Partitioning
<pre class="code">
#set seed for reproducible results
set.seed(1226)
#recursive partitioning-Decision Tree
modDT = train(classe ~ .,method="rpart",data=myTrain)#Faster Computation
</pre>
Run prediction on test partition. View and plot results
<pre class="code">
#Plot 
fancyRpartPlot(modDT$finalModel)
predDT <- predict(modDT, myTest) 
confusionMatrix(predDT, myTest$classe)## <------
</pre>

Result when run on test data partition<br>
<img src="https://98zfdq.by3301.livefilestore.com/y3mdqJXNthr9IOfStTEar-hHuTSX6DQFyu20WeaqkgRNvySw7yjB4pUKWoVmpAoGQsukOLLLHlb1qlQpIv4HF_P1bphkIJ3NfWx8lo1dIIfLWN_08oQpp_gF161Dnoe9lO7G-pEMZI9yEFMfiM0HDv1g17ctyCB0xKVYVyBk1dYbZI?width=480&height=480&cropmode=none" alt="rpartPlot">


<pre class="results">----------------------------------------------------

Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1400  289   43   74   20
         B  397  959  163  481  419
         C  421  256  905  371  280
         D   10   14  257  360   63
         E    4    0    0    0  660

Overall Statistics
                                          
               Accuracy : 0.546           
                 95% CI : (0.5349, 0.5571)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.4284          
 Mcnemar's Test P-Value : < 2.2e-16       

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.6272   0.6318   0.6615  0.27994  0.45770
Specificity            0.9241   0.7693   0.7950  0.94756  0.99938
Pos Pred Value         0.7667   0.3964   0.4053  0.51136  0.99398
Neg Pred Value         0.8618   0.8970   0.9175  0.87034  0.89112
Prevalence             0.2845   0.1935   0.1744  0.16391  0.18379
Detection Rate         0.1784   0.1222   0.1153  0.04588  0.08412
Detection Prevalence   0.2327   0.3083   0.2846  0.08973  0.08463
Balanced Accuracy      0.7757   0.7005   0.7283  0.61375  0.72854
----------------------------------------------------</pre>
</p>
Descision Tree was fast, but accuracy was very low around 50%<br>
You can see how much trouble the model had with predicting the classes looking at the Prediction table.<br>
Result when run on the actual pml-testing data<br>
<pre class="results">
##RESULT -
           ## print(predict(modDT, newdata=nTest))
    #[1] C B B C C C B B A A C B C A B B C B C B
    #Levels: A B C D E
</pre>

<pre class="code">
#rpart with Cross-validation, 
tc <- trainControl("cv",10)
rpart.grid <- expand.grid(.cp=0.2)
set.seed(1010)
modDTCV <- train(classe ~., data=myTrain, method="rpart",trControl=tc,tuneGrid=rpart.grid)

plot(modDTCV$results, main="")
predDTCV <- predict(modDTCV, myTest) 
confusionMatrix(predDTCV, myTest$classe)
</pre>

<pre class="code">
CART 

11776 samples
   52 predictor
    5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Cross-Validated (10 fold) 
Summary of sample sizes: 10600, 10599, 10599, 10598, 10597, 10598, ... 
Resampling results:

  Accuracy   Kappa
  0.2843071  0    

Tuning parameter 'cp' was held constant at a value of 0.2
</pre>
Accuracy is very low, this may be due to the amount of predictors.
<br>
--------------------------<br>
Next I try the Random Forest Model. Classe against all the processed / cleaned data.

<h1>Random Forest Model</h1>

<pre class="code">
##I will try to use Random Forest
#Random Forest Model 
modFitRF <- randomForest(classe ~. , data=myTrain)
modFitRF
predRF <- predict(modFitRF, myTest, type="class") 
confusionMatrix(predRF, myTest$classe)
</pre>

Random Forest Model Results when run on test data partition
<pre class="results">
  Type of random forest: classification
                     Number of trees: 500
No. of variables tried at each split: 7

        OOB estimate of  error rate: 0.7%
Confusion matrix:
     A    B    C    D    E  class.error
A 3345    2    1    0    0 0.0008960573
B   17 2255    7    0    0 0.0105309346
C    0   19 2031    4    0 0.0111976631
D    0    0   24 1905    1 0.0129533679
E    0    0    1    6 2158 0.0032332564
</pre>

<pre class="results">
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 2232    7    0    0    0
         B    0 1504    5    0    0
         C    0    7 1361   11    0
         D    0    0    2 1274    3
         E    0    0    0    1 1439

Overall Statistics
                                          
               Accuracy : 0.9954          
                 95% CI : (0.9937, 0.9968)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9942          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            1.0000   0.9908   0.9949   0.9907   0.9979
Specificity            0.9988   0.9992   0.9972   0.9992   0.9998
Pos Pred Value         0.9969   0.9967   0.9869   0.9961   0.9993
Neg Pred Value         1.0000   0.9978   0.9989   0.9982   0.9995
Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
Detection Rate         0.2845   0.1917   0.1735   0.1624   0.1834
Detection Prevalence   0.2854   0.1923   0.1758   0.1630   0.1835
Balanced Accuracy      0.9994   0.9950   0.9961   0.9950   0.9989
</pre>

<pre class="code">
#Cross Validation, Overfitting.  out-of-sample error is: - 1- .9942 = .0058
myTest$predRight <- predRF==myTest$classe;
qplot(classe, data=myTest, main="Predictions") + facet_grid(predRight ~ .)

# Plot the Random Forest model
plot(modFitRF, uniform=TRUE, main="Error Rate for Random Forest")
legend("topright", colnames(modFitRF$err.rate),col=1:4,cex=0.8,fill=1:4)
</pre>

Plot:<br>

<img src="https://bsdfdq.by3301.livefilestore.com/y3mfGr5E_Rx3FrgD0csPvf82prDMryXklb2gndHNyc8qauf-J7SZmnUh6mX1gDVpmajJjErxmeT8mq1JbepdS2ndkmAXDpCXX-ARO0xCYIniXacW7Jt_zoT4RjcqXe3EBr-axHTeGyOs4JcPR7a6t0WQKQoP0vfNJZdBk_w1ATq17k?width=552&height=325&cropmode=none" alt="errorPlot" style="width:370px; height:202px;">
<pre class="results">
    DT/rpart speed        vs       Random Forest
  user  system elapsed         user  system elapsed 
  10.72    3.16   14.05       26.55    0.06   26.76 
</pre>
Random Forest was slower to computer than DT / rpart, but accuracy was higher about 99%, as expected<br>
<h2>Final Random Forest Model Results</h2>
<pre class="results">
#####################
### PREDICTION ASSIGNMENT SUBMISSION ###
##Run chosen model (Random Forest) and Predict on the pml Testing set of 20 obsevations
##Therefore using the Random Forest Model we can predict 
##the class that the pml test observations fall, i.e(A,B,C,D,E)
print(predict(modFitRF, newdata=nTest))

1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
Levels: A B C D E

#Number of participants in each class. I hope its 20 :)
#predResults = table(predict(modFitRF, newdata=nTest))

A B C D E 
7 8 1 1 3 
</pre>

The majority of the 20 participants either did the curls Exactly according to the specification (A) or
by Throwing their elbows to the front (B).

<p class= cit>
<text style="font-size:12px">
Reference:<br>
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.</text>
</p>
</body>
</html>
