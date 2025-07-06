import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
import h5py
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
data_location = ".\results\spectrograms_example\img_channel_10s\log\disorder"

def train_CNN_disorder(data_location,ch_typ,img_width=32, img_height=32):
    
    img_width, img_height = img_width, img_height
    train_data_dir = "{}\\{}\\train".format(data_location,ch_typ)
    validation_data_dir = "{}\\{}\\val".format(data_location,ch_typ)
    test_data_dir = "{}\\{}\\test".format(data_location,ch_typ)
    
    """Calculate number of samples per class for a modality."""
    a = len(os.listdir("{}\\{}\\train\\B".format(data_location,ch_typ)))
    b = len(os.listdir("{}\\{}\\train\\I".format(data_location,ch_typ)))
    c = len(os.listdir("{}\\{}\\train\\N".format(data_location,ch_typ)))
    d = len(os.listdir("{}\\{}\\train\\Na".format(data_location,ch_typ)))
    e = len(os.listdir("{}\\{}\\train\\Nf".format(data_location,ch_typ)))
    f = len(os.listdir("{}\\{}\\train\\P".format(data_location,ch_typ)))
    g = len(os.listdir("{}\\{}\\train\\Rb".format(data_location,ch_typ)))
    h = len(os.listdir("{}\\{}\\train\\S".format(data_location,ch_typ)))

    a1 = len(os.listdir("{}\\{}\\val\\B".format(data_location,ch_typ)))
    b1 = len(os.listdir("{}\\{}\\val\\I".format(data_location,ch_typ)))
    c1 = len(os.listdir("{}\\{}\\val\\N".format(data_location,ch_typ)))
    d1 = len(os.listdir("{}\\{}\\val\\Na".format(data_location,ch_typ)))
    e1 = len(os.listdir("{}\\{}\\val\\Nf".format(data_location,ch_typ)))
    f1 = len(os.listdir("{}\\{}\\val\\P".format(data_location,ch_typ)))
    g1 = len(os.listdir("{}\\{}\\val\\Rb".format(data_location,ch_typ)))
    h1 = len(os.listdir("{}\\{}\\val\\S".format(data_location,ch_typ)))



    print("Training B = ", a)
    print("Training I = ", b)
    print("Training N = ", c)
    print("Training Na = ", d)
    print("Training Nf = ", e)
    print("Training P = ", f)
    print("Training Rb = ", g)
    print("Training S = ", h)

    print("val B = ", a1)
    print("val I = ", b1)
    print("val N = ", c1)
    print("val Na = ", d1)
    print("val Nf = ", e1)
    print("val P = ", f1)
    print("val Rb = ", g1)
    print("val S = ", h1)
    batch_size = 10
    nb_train_samples = (a+b+c+d+e+f+h+g)/batch_size
    nb_validation_samples = (a1+b1+c1+d1+e1+f1+h1+g1)/batch_size 
    epochs = 100
    
    """Train and evaluate VGG16 model for a single modality."""
    # Data generators
    train_datagen = ImageDataGenerator(
      rescale = 1./255,
      horizontal_flip = True,
      fill_mode = "nearest",
      zoom_range = 0.3,
      width_shift_range = 0.3,
      height_shift_range=0.3,
      rotation_range=30)
    
    val_datagen = ImageDataGenerator(rescale=1./255)

    test_datagen = ImageDataGenerator( rescale = 1./255)

    train_generator = train_datagen.flow_from_directory(
      train_data_dir,
      target_size = (img_height, img_width),
      batch_size = batch_size,
      class_mode = "categorical",shuffle=False)

    validation_generator = val_datagen.flow_from_directory(validation_data_dir,
                                                target_size = (img_height, img_width),
                                                batch_size = batch_size,
                                                class_mode = 'categorical',shuffle=False)
                                                
    test_generator = test_datagen.flow_from_directory(test_data_dir,
                                                target_size = (img_height, img_width),
                                                batch_size = batch_size,
                                                class_mode = 'categorical',shuffle=False)

    # Build and train model
    model = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(img_width, img_height, 3))

    # Creating dictionary that maps layer names to the layers
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    # Getting output tensor of the last VGG layer that we want to include
    x = layer_dict['block5_pool'].output
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(8, activation='softmax')(x)

    # Creating new model. Please note that this is NOT a Sequential() model.
    custom_model = Model(inputs= model.input, outputs=x)
    custom_model.summary()
    # Make sure that the pre-trained bottom layers are not trainable
    for layer in custom_model.layers[:7]:
        layer.trainable = False
    # Do not forget to compile it
    custom_model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                         metrics=['accuracy'])

    checkpoint = ModelCheckpoint("first_level_model\vgg16_{}_disorder.h5".format(ch_typ), monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=20, verbose=1, mode='auto')

    history = custom_model.fit_generator(train_generator,
                                         steps_per_epoch = nb_train_samples,
                                         epochs = epochs,
                                         validation_data = validation_generator,
                                         validation_steps = nb_validation_samples,
                                         callbacks = [checkpoint,early],
                                         verbose = 1)
    
    # Save training history                                    
    hi_df = pd.DataFrame(history.history)
    hi_df.to_csv("first_level_model\hi_df_{}_disorder.csv".format(ch_typ))
    
    # Save model
    custom_model.save('first_level_model\my_full_model_log_{}_disorder.h5'.format(ch_typ))
    custom_model.save_weights('first_level_model\my_model_weights_log_{}_disorder.h5'.format(ch_typ), overwrite=True)
    json_string = custom_model.to_json()
    with open('first_level_model\only_model_{}_disorder.json'.format(ch_typ), 'w') as f:
          f.write(json_string)
          
    # Evaluate
    scores = custom_model.evaluate_generator(train_generator,steps = nb_train_samples, verbose=1)
    print("train_generator: %s: %.2f%%" % (custom_model.metrics_names[1], scores[1]*100))
    scores1 = custom_model.evaluate_generator(validation_generator,steps = nb_validation_samples, verbose=1)
    print("validation_generator: %s: %.2f%%" % (custom_model.metrics_names[1], scores1[1]*100))

    
    #Confution Matrix and Classification Report - get first level probability vector
    Y_pred = custom_model.predict_generator(validation_generator,steps = nb_validation_samples)
    df_test = pd.DataFrame(Y_pred)
    df_test.to_csv("probability vector\{}_val_data_disorder.csv".format(ch_typ))
    lf_test = pd.DataFrame(validation_generator.classes)
    lf_test.to_csv("probability vector\{}_val_label_disorder.csv".format(ch_typ))
    y_pred = np.argmax(Y_pred, axis=1)

    Test_pred = custom_model.predict_generator(train_generator,steps = nb_train_samples)
    df_train = pd.DataFrame(Test_pred)
    df_train.to_csv("probability vector\{}_train_data_disorder.csv".format(ch_typ))
    lf_train = pd.DataFrame(train_generator.classes)
    lf_train.to_csv("probability vector\{}_train_label_disorder.csv".format(ch_typ))
    test_pred = np.argmax(Test_pred, axis=1) 

    Ytest_pred = custom_model.predict_generator(test_generator,steps = nb_validation_samples)
    df_ytest = pd.DataFrame(Ytest_pred)
    df_ytest.to_csv("probability vector\{}_test_data_disorder.csv".format(ch_typ))
    lf_ytest = pd.DataFrame(test_generator.classes)
    lf_ytest.to_csv("probability vector\{}_test_label_disorder.csv".format(ch_typ))
    
    print('Confusion Matrix VAL')
    print(confusion_matrix(validation_generator.classes, y_pred))
    print('Classification Report VAL')
    print(classification_report(validation_generator.classes, y_pred))

    print('Confusion Matrix  TRAIN')
    print(confusion_matrix(train_generator.classes, test_pred))
    print('Classification Report TRAIN')
    print(classification_report(train_generator.classes, test_pred))



def main():
    """Train VGG16 models for each modality (EEG, ECG, EMG)."""
    MODALITIES = ['EEG', 'ECG', 'EMG']  
    for ch_typ in MODALITIES:
        print(f"\nTraining model for modality: {ch_typ}")
        train_CNN_disorder(data_location,ch_typ,32,32)

if __name__ == "__main__":
    main()











































