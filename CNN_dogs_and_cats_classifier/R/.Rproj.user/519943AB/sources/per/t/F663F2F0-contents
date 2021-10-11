#Classifer dogs and cats usin CNN transfering learning
#It is the same problem as Classifier_dogs_and_cats.R, but, using transfering learning to achieve more accuracy.
#Importing libraries and generating dataset
require(keras)
train_dir <- file.path("./train")
valid_dir <- file.path("./validation")
test_dir <- file.path("./test")

datagen_train <- image_data_generator(
  rescale = 1 / 255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = T,
  fill_mode = "nearest"
)

datagen_valid <- image_data_generator(
  rescale = 1 / 255
)

datagen_test <- image_data_generator(
  rescale = 1 / 255
)

generator_train <- flow_images_from_directory(
  train_dir,
  datagen_train,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

generator_valid <- flow_images_from_directory(
  valid_dir,
  datagen_valid,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

generator_test <- flow_images_from_directory(
  test_dir,
  datagen_test,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

#VGG16
conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = F,
  input_shape = c(150, 150, 3)
)

model_vgg <- keras_model_sequential()%>%
  conv_base%>%
  layer_flatten()%>%
  layer_dense(256, activation = "relu")%>%
  layer_dense(1, activation = "sigmoid")
model_vgg
model_vgg %>% compile(loss = "binary_crossentropy",
                      optimizer = optimizer_adam(lr = 2e-5), 
                      metrics = list("acc"))
cb_list = list(callback_early_stopping(patience = 25,
                                       restore_best_weights = T),
               callback_model_checkpoint("best_vgg16R.h5"))
freeze_weights(conv_base) #Avoid train weights from vgg convolutional base
history_vgg <- model_vgg %>% fit_generator(generator_train, epoch = 1000,
                                        validation_data = generator_valid,
                                        callbacks = cb_list,
                                        steps_per_epoch = 100,
                                        validation_steps = 50)
plot(history_vgg)
rm(model_vgg)
k_clear_session()

best_model_vgg <- load_model_hdf5("best_vgg16R.h5")
score <- best_model_vgg %>% evaluate_generator(generator_test, steps = 50)
#I run this model in Google colab beacause of my computational power
#This model obtained 97,10% accuracy from test dataset