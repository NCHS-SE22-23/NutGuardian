from pathlib import Path
from model import barbecue, munch, taste_test_results, learn_recipe
from data_creator import prepare_ingredients
from dataset_manipulation import chop_vegetables

#USE TF LITE FOR RASPBERRY PI

#Downloading the data
if not Path('images').exists():
    prepare_ingredients()

#Splitting the datasets
data = Path('images')
train_ds, val_ds = chop_vegetables(data)

#Creating the model
model = barbecue()

#Compiling and training the model
history = munch(model, train_ds, val_ds)

#Plotting accuracy
taste_test_results(history)

#Saving the model's weights
learn_recipe(model, 2)