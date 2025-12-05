from tensorflow.keras.datasets import mnist #inports the dataset
from tensorflow.keras.models import Sequential #builds the layers
from tensorflow.keras.layers import Conv2D,Flatten,Dense,Reshape 
#reshape-adds the channel dimension
#Conv2D-scans the image using filters
#Flatten-turns images tov grids with numbers
#Dense-Makes the final classification decisions
from tensorflow.keras.optimizers import Adam
#Adam-smart weight adjuster

#Load data
(x_train,y_train),(x_test,y_test)=mnist.load_data()
#x_train->images y_train->correct digits(labels)

#Normalise images(divide by 255 pixels to get values btw 0.0-1.0)
x_train=x_train/255.0
x_test=x_test/255.0

#CNN
model = Sequential([
    Reshape((28,28,1), input_shape=(28,28)),#last digit is the channel (here 1)
    Conv2D(16,(3,3), activation="relu"),
    Conv2D(16,(3,3), activation="relu"),#its number of filters followed by each filter size and relu is -ves to 0
    Flatten(),#map grids to single vector
    Dense(10, activation="softmax")
])


model.compile(
    optimizer=Adam(),#determine the weights
    loss="sparse_categorical_crossentropy", #measures how wrong the predictions are
    metrics=["accuracy"] #tracks percentage of correct predictions
)

model.fit(x_train,y_train, epochs=6,validation_split=0.1) #epoch number of times it trains #validation split reserves 10% of data to check overfitting

loss,acc=model.evaluate(x_test,y_test)

print("FINAL TEST ACCURACY:",acc)


