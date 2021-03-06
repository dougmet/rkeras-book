library(keras)

imdb <- dataset_imdb(num_words = 10000)
saveRDS(imdb, "book/imdb.rds")
imdb <- readRDS("book/imdb.rds")

# Load training sets

train_data <- imdb$train$x
train_labels <- imdb$train$y
test_data <- imdb$test$x
test_labels <- imdb$test$y

# Look at the index
word_index <- dataset_imdb_word_index()
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index
head(reverse_word_index)

decoded_review <- sapply(train_data[[1]], 
                         function(index) {
                           word <- if(index >= 3) reverse_word_index[[as.character(index - 3)]]
                           if(!is.null(word)) word else "?"
                         })

vectorize_sequences <- function(sequences, dimension = 10000) {
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  
  for (i in seq_along(sequences)) {
    results[i, sequences[[i]]] <- 1
  }
  
  results
}

x_train <- vectorize_sequences(train_data)
y_train <- as.numeric(train_labels)

model <- keras_model_sequential() %>%
  layer_dense(units = 16, input_shape = c(10000), activation = "relu") %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

val_indices <- 1:10000

x_val <- x_train[val_indices, ]
partial_x_train <- x_train[-val_indices,]

y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]


history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

as.data.frame(history)

