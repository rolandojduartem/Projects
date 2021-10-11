#Classify dogs and cats using a convutional neural network
require(keras)
#Creting directories
train_dir <- file.path("train")
valid_dir <- file.path("validation")
test_dir <- file.path("test")

#Feeding generators with images
train_datagen <- image_data_generator(rescale = 1 / 255)
valid_datagen <- image_data_generator(rescale = 1 / 255)
test_datagen <- image_data_generator(rescale = 1 / 255)

#Creating generators

train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

valid_generator <- flow_images_from_directory(
  valid_dir,
  valid_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)
test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

#Model architecture
model <- keras_model_sequential()%>%
  layer_conv_2d(32, c(3, 3), activation = "relu", input_shape = c(150, 150, 3))%>%
  layer_max_pooling_2d(c(2, 2))%>%
  layer_conv_2d(64, c(3, 3), activation = "relu")%>%
  layer_max_pooling_2d(c(2, 2))%>%
  layer_conv_2d(128, c(3, 3), activation = "relu")%>%
  layer_max_pooling_2d(c(2, 2))%>%
  layer_conv_2d(128, c(3, 3), activation = "relu")%>%
  layer_max_pooling_2d(c(2, 2))%>%
  layer_flatten()%>%
  layer_dense(512, activation = "relu")%>%
  layer_dense(1, activation = "sigmoid")

summary(model)
#Compiling model
model %>% compile(loss = "binary_crossentropy",
                  optimizer = optimizer_rmsprop(lr = 1e-4),
                  metrics = list("acc"))
#Fitting model
history <- model %>% fit_generator(train_generator,
                         steps_per_epoch = 100,
                         epochs = 20,
                         validation_data = valid_generator,
                         validation_steps = 50)
# Saving model 
model %>% save_model_hdf5("modelR.h5")
model <- load_model_hdf5("modelR.h5")
#Displaying the fitting data
plot(history)
model %>% evaluate_generator(test_generator, steps = 50)
rm(model)
k_clear_session()



#____________________________________________________________________


#Data augmentation
#Using image generator to create new image data
datagen <- image_data_generator(
  rescale = 1 / 255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  zoom_range = 0.2,
  shear_range = 0.2,
  horizontal_flip = T,
  fill_mode = "nearest"
)
train_auggen <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)
valid_auggen <- flow_images_from_directory(
  valid_dir,
  valid_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)
test_auggen <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

k_clear_session()

#Creating new model
model_aug <- keras_model_sequential()%>%
  layer_conv_2d(32, c(3, 3), activation = "relu", input_shape = c(150, 150, 3))%>%
  layer_max_pooling_2d(c(2, 2))%>%
  
  layer_conv_2d(64, c(3, 3), activation = "relu")%>%
  layer_max_pooling_2d(c(2, 2))%>%
  
  layer_conv_2d(128, c(3, 3), activation = "relu")%>%
  layer_max_pooling_2d(c(2, 2))%>%
  
  layer_conv_2d(128, c(3, 3), activation = "relu")%>%
  layer_max_pooling_2d(c(2, 2))%>%
  
  layer_flatten()%>%
  layer_dense(512, activation = "relu")%>%
  layer_dense(1, activation = "sigmoid")
summary(model_aug)
#Compiling new model
model_aug %>% compile(loss = "binary_crossentropy",
                      optimizer = optimizer_rmsprop(lr = 1e-4),
                      metrics = c("acc"))
#Creating callbacks
callbacks_list = list(callback_early_stopping(patience = 10,
                                              restore_best_weights = T),
                      callback_model_checkpoint("best_aug_modelR.h5",
                                                save_best_only = T))
#Fitting model
history <- model_aug %>% fit_generator(train_auggen,
                                       steps_per_epoch = 100,
                                       epochs = 100,
                                      validation_data = valid_auggen,
                                      validation_steps = 50,
                                      callbacks = callbacks_list)

best_model_aug <- load_model_hdf5("best_aug_modelR.h5")
best_model_aug %>% evaluate_generator(test_auggen, steps = 50)


rm(model_aug)
k_clear_session()
