original_dataset_dir <- "book/dogscats/train"
base_dir <- "book/dogscats_small"
dir.create(base_dir)
train_dir <- file.path(base_dir, "train")
dir.create(train_dir)
validation_dir <- file.path(base_dir, "validation")
dir.create(validation_dir)
test_dir <- file.path(base_dir, "test")
dir.create(test_dir)
train_cats_dir <- file.path(train_dir, "cats")
dir.create(train_cats_dir)
train_dogs_dir <- file.path(train_dir, "dogs")
dir.create(train_dogs_dir)
validation_cats_dir <- file.path(validation_dir, "cats")
dir.create(validation_cats_dir)
validation_dogs_dir <- file.path(validation_dir, "dogs")
dir.create(validation_dogs_dir)
test_cats_dir <- file.path(test_dir, "cats")
dir.create(test_cats_dir)
test_dogs_dir <- file.path(test_dir, "dogs")
dir.create(test_dogs_dir)
fnames <- paste0("cat.", 1:1000, ".jpg")
file.rename(file.path(original_dataset_dir, fnames),
          file.path(train_cats_dir, fnames))
fnames <- paste0("cat.", 1001:1500, ".jpg")
file.rename(file.path(original_dataset_dir, fnames),
          file.path(validation_cats_dir, fnames))
fnames <- paste0("cat.", 1501:2000, ".jpg")
file.rename(file.path(original_dataset_dir, fnames),
          file.path(test_cats_dir, fnames))
fnames <- paste0("dog.", 1:1000, ".jpg")
file.rename(file.path(original_dataset_dir, fnames),
          file.path(train_dogs_dir, fnames))
fnames <- paste0("dog.", 1001:1500, ".jpg")
file.rename(file.path(original_dataset_dir, fnames),
          file.path(validation_dogs_dir, fnames))
fnames <- paste0("dog.", 1501:2000, ".jpg")
file.rename(file.path(original_dataset_dir, fnames),
          file.path(test_dogs_dir, fnames))
