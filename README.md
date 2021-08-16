# Food Classification
## About
`food101` dataset consists of 101 food categories, with 101,000 images. The training dataset has 75,750 images and the test dataset has 25,250 images. Each food class contains 1000 images.

## Exploratory Data Analysis

`
Number of classes:  101 
`

`
Classes:  ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']
`

A few samples of how our dataset looks like:
![data_preview](https://user-images.githubusercontent.com/53920732/129536145-5641bc91-49cf-47c6-a7d4-8240d2973a37.png)

## Preprocessing Data

Since we've downloaded the data from TensorFlow Datasets, there are a couple of preprocessing steps we have to take before our data is ready to model.
Our data is currently:
- In `unint8` data type
- Comprises of tensors of different shape (different sized images)
- Not scaled (Since we're working on an images dataset, the max value of our data is 255 and the min value is 0.)

What we want our data to be:
- In `float32` data type
- All image tensors of same shape (since we're going to turn our data into batches, it requires that all tensors be of same shape).
- Scaled (Neural Networks prefer data to be scaled. To get better results, we need to normalize our data i.e turn it into range of (0,1).  We can normalize our data by dividing it by the max value i.e. dividing it by 255.)


We'll create a `preprocess_image()` function which will take image and labels as input and:
- Resize an input image tensor to a specific size
- Convert an input image tensor' datatypr to `tf.float32`. 
- Since we're going to be using the pretrained EfficientNetBX model, it already has rescaling built-in. Therefore we don't need to rescale it in our function.

```Python
def preprocess_image(image, label, image_shape=224):
  """
  Converts image datatype from 'uint8' -> 'float32' and reshapes image to
  [image_shape, image_shape, color_channels]
  """

  image  = tf.image.resize(image, [image_shape, image_shape]) #reshape
  image = tf.cast(image, tf.float32) #typecast
  return image, label
```
When we apply the above function to an image:

```
Before Preprocessing:  
Image Shape:  (384, 512, 3) 
Image dtype:  <dtype: 'uint8'>
After Preprocessing:  
Image Shape:  (224, 224, 3) 
Image dtype:  <dtype: 'float32'>
```
![after preprocessing](https://user-images.githubusercontent.com/53920732/129557283-b2946162-b87a-427c-abb6-672cdd88b183.png)


## Batch and Prepare Datasets

When working with large datasets, its not possible to train the entire dataset at once as it may not fit into the memory and even if it does, the entire process will be very slow. And therefore what we do instead is take the dataset and break it into batches and train our model batch by batch. We're also going to shuffle our dataset to generalize our model.

We will turn our data from 101,000 image tensors and labels into batches of 32 image and label pairs. We're going to use methods from `tf.dataAPI`.
- `map()`: map a predefined function to a target dataset 
- `shuffle()`: randomly shuffle elements of the target dataset
- `batch()`: turn elements of a target dataset into batches of `batch_size`
- `cache()`: caches elements in a target dataset and saves loading rime.

In other words, we're going to map our preprocessing function across our training dataset then shuffle the elements before batching them together and making sure to prepare new batches (prefetch) whilst the model is looking at the current batch. This makes the process faster.

```Python
#Map preprocessing function to the training data and parallelize
train_data = train_data.map(
    map_func = preprocess_image,
    num_parallel_calls = tf.data.AUTOTUNE #autotune the number of processors used (utilize as many processor as you can find)
    )

#Shuffle train_data and turn it into batches and prefetch
train_data = train_data.shuffle(
    buffer_size = 1000 #how many elements to shuffle at a single time
    ).batch(
        batch_size = 32 #number of elements in a batch
        ).prefetch(
            buffer_size = tf.data.AUTOTUNE
        )

#Map preprocessing function to the test data and parallelize
test_data = test_data.map(map_func = preprocess_image,
                          num_parallel_calls = tf.data.AUTOTUNE)

#Turn test data into batches (no need to shuffle)
test_data = test_data.batch(
    batch_size = 32).prefetch(
        buffer_size = tf.data.AUTOTUNE
    )
  ```
    
## Modelling Callbacks
    
Since we're going to be training our model on a large amount of data and training takes a long time, we're going to set up some modelling callbacks so that we can track our model's training logs and make sure our model is being checkpointed after various training milestones.

We'll use `tf.keras.callbacks.TensorBoard()` as it allows us to keep track of our model's training history so we can inspect it later. `tf.keras.callbacks.ModelCheckpoint()` saves our model's progress at various intervals so that we can load it and reuse it later without having to retain it. This is also helpful when we fine-tune our model at a particular epoch and revert back to a previous state if fine-tuning offers no benefits.

```Python
#Function to create tensorboard callback
def create_tensorboard_callback(dir_name, experiment_name):
  """
  Creates a TensorBoard callback instance to store log files.
  Stores log files with the filepath:
    "dir_name/experiment_name/current_datetime/"
  Args:
    dir_name: target directory to store TensorBoard log files
    experiment_name: name of experiment directory
  """
  log_dir = dir_name + '/' + experiment_name + '/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
  tensor_board_callback = tf.keras.callbacks.TensorBoard(
      log_dir = log_dir
  )
  print(f'Saving TensorBoard log file to: {log_dir}')
  return tensor_board_callback
```

```Python

#Create ModelCheckpoint callback to save model's progress during training
#We're only going to save the model's weights as it is a lot faster than storing the whole model
checkpoint_path = 'model_checkpoint/cp.ckpt'
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_path,
    monitor = 'val_accuracy', #save model weights with validation accuracy
    save_best_only = True, #only save best weights
    save_weights_only = True, #only save model weights
    verbose = 0)
```
## Feature Extraction Model

Since our dataset is quite large, we're going to fine-tune an existing pretrained model. But first we need to set up a feature-extraction model.
When transfer-learning, the typical order is:
- Build a feature extraction model ( replace the top few layers of a pretrained model).
- Train for a few epochs with lower (base) layers frozen.
- Fine-tune if necessary with multiple layers unfrozen.

To build the feature extraction model:
- We're going to use `EfficientNetB0` from `tf.keras.applications` pre-trained on ImageNet as our base model. 
- First we will download without top layers using `include_top=False` parameter so that we can create our own output layers.
- We'll freeze the base model layer so that we can use the pre-trained patterns the base model has found on ImageNet.
- We will then put together the input, base model, pooling layers and output layer in a functional model.
- Then compile the functional model using Adam optimizer and sparse categorical crossentropy as the loss function since our labels aren't one-hot encoded.
- Then fit the model using TensorBoarf and ModelCheckpoint callbacks.

```Python
#Download base model and freeze underlying layers

input_shape = (224, 224, 3)
#Create base model
base_model = tf.keras.applications.EfficientNetB0(include_top = False)
#Freeze base model layers
base_model.trainable = False 

#Create functional model
inputs  = tf.keras.layers.Input(shape=input_shape, name='input_layer')
#Set base model to inference model only
x = base_model(inputs, training=False) 
x = tf.keras.layers.GlobalAveragePooling2D(name='pooling_layer')(x)
outputs = tf.keras.layers.Dense(len(classes), activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

#Compile the model
model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = tf.keras.optimizers.Adam(),
    metrics = ['accuracy']
)
```
```Python
model.summary()
```

<img width="327" alt="featuremodel_summary" src="https://user-images.githubusercontent.com/53920732/129557889-daf3bcd9-97b5-4337-9f51-55638c51a06d.PNG">

## Training
We fit the model for 3 epochs. That should be enough for our top layer to adjust their weights to the dataset. 

![featurehistory](https://user-images.githubusercontent.com/53920732/129558118-939da0f8-41f7-4e17-9c85-8e56072b52bf.png)


## Fine-tuning the model

We get an accuracy of around `71%` after training our feature extraction model on three epochs. We can increase the performance of our model by fine-tuning it. When fine-tuning, the traditional workflow is to freeze the pre-trained base model and then train only the output layers for a few iterations so that the weights can be updated inline with the custom data. And then unfreeze a number or all of the layers in the base model and continue training until the model stops imporving.


When we first created the feature extraction model, we froze all of the layers in the base model by setting `base_model.trainable = False`. But since we loaded in our model from file, we can set `base_model.trainable = True`.


```Python
#Set all the layers to trainable
for layer in loaded_model.layers:
  layer.trainable = True 
  print(layer.name, layer.trainable)
```

## Callbacks for Fine-Tuning

We will also use `tf.keras.callbacks.EarlyStopping()` as it monitors a specific model performance metric and when it stops improving for a specific number of epochs, it automatically stops the training of our model.

Since we're using `EarlyStopping` callback combined with the `ModelCheckpoint` callback saving the best performing model automatically, we could keep our model training for an unlimited number of epochs until it stops improving.

```Python
#Setup EarlyStopping callback to stop training if the model's val_loss doesnt improve for 3 epochs
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss', #monitor validation loss metric
    patience = 3 #if val loss doesnt decrese for 3 epoch in a row, stop training
)

#Create ModelCheckpoint call to save best model during fine-tuning
checkpoint_path = 'fine_tune_checkpoints/'
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_path,
    monitor = 'val_loss',
    save_best_only = True)
```

We will also use the `ReduceLROnPlateau` callback. As the name suggests, it helps in tuning the learning rate. It monitors a specific metric and when the metric stops improving, it reduces the learning rate by a specific factor. When our model gets closer to its ideal performance, i.e when it starts to converge, we want the size of steps to get smaller and smaller. We will monitor the validation loss, and when it stops improving for two or more epochs, we will reduce the learning rate by a fact of 5.
We will also set the minimum learning rate so as to make sure it doenst get too low.

```Python
#Create learning rate reduction callback
reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = 'val_loss', #monitor the validation loss
    factor = 0.2, #multiply the learning rate by 0.2
    verbose = 1, #print when the learning rate changes
    min_lr = 1e-7 #mimimum learning rate
)
```

Now we will fit the model for 100 epochs. Since we're using `EarlyStopping` callback. Since we've set all the layers in the base model to be trainable,fitting the model will fine-tune all of the pre-trained weights in the base model on the whoel food101 dataset. 


```Python
#Start fine-tuning
history_fine_tune = loaded_model.fit(
    train_data,
    epochs = 100, #fine tune for max 100 epochs
    steps_per_epoch = len(train_data),
    validation_data = test_data,
    validation_steps = int(0.15 * len(test_data)), #validate on only 15% of the data
    callbacks = [create_tensorboard_callback('training_logs', 'model_fine_tuning'), #track training logs 
                 model_checkpoint, #save only the best model during training
                 early_stopping, #stop model after 3 epochs of no improvements
                 reduce_learning_rate #reduce learning rate after 3 epochs of no improvement
    ]
)
```

![finetunehistory](https://user-images.githubusercontent.com/53920732/129558746-f8e663b2-58f6-45d6-906d-3e96b13d16d9.png)


We get an accuracy of around `78%` after fine-tuning our model. Training accuracy seems to be steadily increasing, but validation accuracy seems to have peaks and valleys, but overall it does increase. This might be a sign of some overfitting, but definitely not a large amount of overfitting.
