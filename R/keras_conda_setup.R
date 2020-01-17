####  2020 Attempt  ####
# The following installation steps were used to achieve succesful installation of keras and tensorflow for mac
# rkeras2020 is a conda environment initiated with python 3.6

####  Github installs  ####
devtools::install_github("rstudio/tensorflow")
devtools::install_github("rstudio/keras")

#Switch to conda env we want to use
reticulate::use_condaenv("rkeras2020")
reticulate::py_discover_config() #Check that python library setup was succesful

#Installing tensorflow
tensorflow::install_tensorflow()
tensorflow::tf_config()


#Packages
library(keras)


####  Load MNST Test Data  ####
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y


# reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test  <- array_reshape(x_test, c(nrow(x_test), 784))

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)


# Defining the model
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')


summary(model)

#Compile the model with appropriate loss function, optimizer, and metrics
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)


#Training and evaluation
history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)

# history object includes loss and accuracy metrics
plot(history)

#Evaluate the model's performance on the test data
model %>% evaluate(x_test, y_test)

#Generate predictions on new data
model %>% predict_classes(x_test)
