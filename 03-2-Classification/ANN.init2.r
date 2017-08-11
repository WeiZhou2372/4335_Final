needs(rnn)
data = source("./03-2-Classification/data.getReady.r", local = T)$value;rm(b)

train.data = data[1:1000, 1:5]
test.data = data[1001:1200, 1:5]

X1 = train.data$settle
X2 = train.data$score * 1000

Y = train.data$y

X1 <- int2bin(X1,16)
X2 <- int2bin(X2,16)
Y  <- int2bin(Y,16)

X <- array( c(X1,X2), dim=c(dim(X1),2) )
Y <- array( Y, dim=c(dim(Y),1) ) 

model <- trainr(Y=Y[,dim(Y)[2]:1,,drop=F], # we inverse the time dimension
                X=X[,dim(X)[2]:1,,drop=F], # we inverse the time dimension
                learningrate   =  0.3,
                numepochs = 15,
                hidden_dim     = 5)


plot(colMeans(model$error),type='l',
     xlab='epoch',
     ylab='errors'                  )
# create sample inputs
X1 = sample(0:127, 5000, replace=TRUE)
X2 = sample(0:127, 5000, replace=TRUE)

# create sample output
Y <- X1 + X2

# convert to binary
X1 <- int2bin(X1)
X2 <- int2bin(X2)
Y  <- int2bin(Y)

# Create 3d array: dim 1: samples; dim 2: time; dim 3: variables.
X <- array( c(X1,X2), dim=c(dim(X1),2) )
Y <- array( Y, dim=c(dim(Y),1) ) 