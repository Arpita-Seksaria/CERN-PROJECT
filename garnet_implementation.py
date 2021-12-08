import setGPU
import os
import h5py
import glob
import numpy as np
import matplotlib.pyplot as plt
#from generatorGNN import DataGenerator
import sys
import numpy as np
from tensorflow.keras import metrics
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.optimizers import SGD, Adam
from qkeras.qlayers import QDense, QActivation
try:
    import tensorflow.keras as keras

except ImportError:
    import keras


K = keras.backend

target = np.array([])
jetList = np.array([])
# training                                                                                                                                                                                                  
inputTrainFiles = glob.glob("/eos/project/d/dshep/hls-fml/NEWDATA/*150p*h5")
cut_train = int(len(inputTrainFiles)/3.)
inputValFiles = inputTrainFiles[:cut_train]
inputTrainFiles = inputTrainFiles[cut_train:]
inputTestFiles = glob.glob("/eos/project/d/dshep/hls-fml/NEWDATA/VALIDATION/*150p*h5")

from garnet import  GarNet

vmax = 8
quantize = True
x = keras.layers.Input(shape=(vmax, 3))
n = keras.layers.Input(shape=(1,), dtype='uint16')
inputs = [x, n]
total_bits = int(sys.argv[1])
int_bits = 0
x = keras.layers.BatchNormalization()(x)
v = GarNet(20, 16, 12, simplified=True, collapse='mean', input_format='xn', 
      output_activation=None, name='gar_1', 
      quantize_transforms=quantize,
           total_bits=total_bits, int_bits=int_bits)([x, n])
#v = GarNet(20, 16, 12, simplified=True, collapse='mean', input_format='xn', 
               # output_activation='relu', name='gar_1', quantize_transforms=quantize)([x, n])
if quantize == True:
    v = QActivation(activation='quantized_relu(%i, %i)'%(total_bits, int_bits))(v)
    
    v = QDense(16, kernel_quantizer='quantized_bits(%i, %i, alpha=1.0)'%(total_bits, int_bits), bias_quantizer='quantized_bits(%i,%i,alpha=1.0)'%(total_bits, int_bits))(v)
    v = QActivation(activation='quantized_relu(%i, %i)'%(total_bits, int_bits))(v)
    v = QDense(16, kernel_quantizer='quantized_bits(%i, %i, alpha=1.0)'%(total_bits, int_bits), bias_quantizer='quantized_bits(%i,%i,alpha=1.0)'%(total_bits, int_bits))(v)
    v = QActivation(activation='quantized_relu(%i, %i)'%(total_bits, int_bits))(v)
    v = QDense(16, kernel_quantizer='quantized_bits(%i, %i, alpha=1.0)'%(total_bits, int_bits), bias_quantizer='quantized_bits(%i,%i,alpha=1.0)'%(total_bits, int_bits))(v)
    v = QActivation(activation='quantized_relu(%i, %i)'%(total_bits, int_bits))(v)
    v = QDense(5, kernel_quantizer='quantized_bits(%i, %i, alpha=1.0)'%(total_bits, int_bits),bias_quantizer='quantized_bits(%i, %i, alpha=1.0)'%(total_bits, int_bits))(v)

    outLayer = keras.layers.Activation(activation='softmax',name="softmax")(v)
else:
    v = keras.layers.Activation(activation="relu")(v)
    v = keras.layers.Dense(16)(v)
    v = keras.layers.Activation(activation="relu")(v)
    v = keras.layers.Dense(16)(v)
    v = keras.layers.Activation(activation="relu")(v)
    v = keras.layers.Dense(16)(v)
    v = keras.layers.Activation(activation="relu")(v)
    v = keras.layers.Dense(5)(v)
    outLayer = keras.layers.Activation(activation='softmax',name="softmax")(v)

#outLayer = keras.layers.Dense(5, activation='softmax')(v)
    
model = keras.Model(inputs=inputs, outputs=outLayer)
optim = Adam(learning_rate=0.0002)
model.compile(loss='categorical_crossentropy', optimizer=optim)
model.summary()

batch_size = 512
n_epochs = 150

#my_batch_per_file = int(10000/batch_size)
#myTrainGen = DataGenerator("TRAINING", inputTrainFiles, batch_size, my_batch_per_file, vmax)
#myValGen = DataGenerator("VALIDATION", inputValFiles, batch_size, my_batch_per_file, vmax)

X_1 = np.array([])
Y_1 = np.array([])
for fileName in inputTrainFiles:
    print(fileName)
    f = h5py.File(fileName, "r")
    myX = np.array(f.get('jetConstituentList'))
   # myX = myX[:,:,[5,7,10]]
    myX = myX[:,:,[5,8,11]] 
    myY = np.array(f.get('jets')[0:,-6:-1])
    X_1 = np.concatenate((X_1, myX), axis=0) if X_1.size else myX
    Y_1 = np.concatenate((Y_1, myY), axis=0) if Y_1.size else myY

njet = X_1.shape[0]
nconstit = X_1.shape[1]
nfeat = X_1.shape[2]

print('Shape of jetConstituent =',X_1.shape)
print('Number of jets =',njet)
print('Number of constituents =',nconstit)
print('Number of features =',nfeat)

print('Pt order of jetConstituent =',X_1[0,:,0])
# Filter out constituents with Pt<2GeV                                                                                                                                                                      
Ptmin =2.
constituents = np.zeros((njet, nconstit, nfeat) , dtype=np.float32)
ij=0
max_constit=0
for j in range(njet):
    ic=0
    for c in range(nconstit):
        if ( X_1[j,c,0] < Ptmin ):
            continue
        constituents[ij,ic,:] = X_1[j,c,:]
        ic+=1
    if (ic > 0):
        if ic > max_constit: max_constit=ic
        Y_1[ij,:]=Y_1[j,:] # assosicate the correct target a given graph                                                                                                                              
        ij+=1
# Resizes the jets constituents and target arrays                                                                                                                                                          
X_1 = constituents[0:ij,0:max_constit,:]
Y_1 = Y_1[0:ij,:]
del constituents

X_train, y_train = shuffle(X_1, Y_1)
X_train = X_train[:,:vmax,:]
V_train =  np.ones((X_train.shape[0],1))*vmax
#X_train[:,:,0] = X_train[:,:,0]/1600.

X_2 = np.array([])
Y_2 = np.array([])
for fileName in inputValFiles:
    print(fileName)
    f =h5py.File(fileName, "r")
    myX = np.array(f.get('jetConstituentList'))
   #myX = myX[:,:,[5,7,10]]
    myX = myX[:,:,[5,8,11]]
    myY = np.array(f.get('jets')[0:,-6:-1])
    X_2 = np.concatenate((X_2, myX), axis=0) if X_2.size else myX
    Y_2 = np.concatenate((Y_2, myY), axis=0) if Y_2.size else myY

njet = X_2.shape[0]
nconstit = X_2.shape[1]
nfeat = X_2.shape[2]

print('Shape of jetConstituent =',X_2.shape)
print('Number of jets =',njet)
print('Number of constituents =',nconstit)
print('Number of features =',nfeat)

print('Pt order of jetConstituent =',X_2[0,:,0])

# Filter out constituents with Pt<2GeV                                                                                                                                                                     \
Ptmin =2.
constituents = np.zeros((njet, nconstit, nfeat) , dtype=np.float32)
ij=0
max_constit=0
for j in range(njet):
    ic=0
    for c in range(nconstit):
        if ( X_2[j,c,0] < Ptmin ):
            continue
        constituents[ij,ic,:] = X_2[j,c,:]
        ic+=1
    if (ic > 0):
        if ic > max_constit: max_constit=ic
        Y_2[ij,:]=Y_2[j,:] # assosicate the correct target a given graph                                                                                                                                    
        ij+=1
# Resizes the jets constituents and target arrays                                                                                                                                                          \
X_2 = constituents[0:ij,0:max_constit,:]
Y_2 = Y_2[0:ij,:]
del constituents
X_val, y_val = shuffle(X_2, Y_2)
X_val = X_val[:,:vmax,:]
V_val = np.ones((X_val.shape[0],1))*vmax
#X_val[:,:,0] = X_val[:,:,0]/1600.

history = model.fit((X_train, V_train), y_train, epochs=n_epochs, batch_size=512,
                    validation_data = ((X_val, V_val), y_val),
                    verbose=1,
                    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=1),
                                 ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1),
                                 TerminateOnNaN()])

model.save('/eos/user/a/arseksar/output_paper/GNN_model_%ibit.h5'%(total_bits))
# other outputs                                                                                                                                                                                             
X_test = np.array([])
y_test = np.array([])
for fileIN in inputTestFiles:
    f = h5py.File(fileIN, "r")
    myFeatures = np.array(f.get('jetConstituentList'))
    # myFeatures = myFeatures[:,:,[5,7,10]]
    myFeatures = myFeatures[:,:,[5,8,11]] 
    mytarget = np.array(f.get('jets')[0:,-6:-1])
    X_test = np.concatenate([X_test, myFeatures], axis=0) if X_test.size else myFeatures
    y_test = np.concatenate([y_test, mytarget], axis=0) if y_test.size else mytarget
njet = X_test.shape[0]
nconstit = X_test.shape[1]
nfeat = X_test.shape[2]

print('Shape of jetConstituent =',X_test.shape)
print('Number of jets =',njet)
print('Number of constituents =',nconstit)
print('Number of features =',nfeat)

print('Pt order of jetConstituent =',X_test[0,:,0])
# Filter out constituents with Pt<2GeV                                                                                                                                                                     \
Ptmin =2.
constituents = np.zeros((njet, nconstit, nfeat) , dtype=np.float32)
ij=0
max_constit=0
for j in range(njet):
    ic=0
    for c in range(nconstit):
        if ( X_test[j,c,0] < Ptmin ):
            continue
        constituents[ij,ic,:] = X_test[j,c,:]
        ic+=1
    if (ic > 0):
        if ic > max_constit: max_constit=ic
        y_test[ij,:]=y_test[j,:] # assosicate the correct target a given graph                                                                                                                                    
        ij+=1
# Resizes the jets constituents and target arrays                                                                                                                                                          
X_test = constituents[0:ij,0:max_constit,:]
y_test = y_test[0:ij,:]
del constituents



f.close()
X_test = X_test[:,:vmax,:]
#X_test[:,:,0] = X_test[:,:,0]/1600.
V_test = np.ones((X_test.shape[0],1))*vmax
preds = model.predict((X_test, V_test), batch_size=1000)
outFile = h5py.File("/eos/user/a/arseksar/output_paper/GNN_out_%ibit.h5"%(total_bits), "w")


outFile.create_dataset('loss', data=history.history['loss'], compression='gzip')
outFile.create_dataset('val_loss', data=history.history['val_loss'], compression='gzip')
outFile.create_dataset('preds', data = np.array(preds),  compression='gzip')
outFile.create_dataset('target', data= np.array(y_test), compression='gzip')
outFile.create_dataset('input', data= np.array(X_test), compression='gzip')
outFile.close()

def regression_loss(y_true, y_pred):
    with K.name_scope('regression_loss'):
        y_true /= 100. # because our data is max 100 GeV

        return K.mean(K.square((y_true - y_pred) / y_true), axis=-1)

classification_loss = 'binary_crossentropy'

