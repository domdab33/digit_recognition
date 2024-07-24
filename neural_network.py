import numpy as np
from numpy.random import randn
import idx2numpy
import matplotlib.pyplot as plt

#MNIST DATA
train_images = 'train-images.idx3-ubyte'
train_labels = 'train-labels.idx1-ubyte'
test_images = 't10k-images.idx3-ubyte'
test_labels = 't10k-labels.idx1-ubyte'

#data as numpy
train_images_array = idx2numpy.convert_from_file(train_images)
train_labels_array = idx2numpy.convert_from_file(train_labels)
test_images_array = idx2numpy.convert_from_file(test_images)
test_labels_array = idx2numpy.convert_from_file(test_labels)

#normalized images
train_images_array = train_images_array/255.0
test_images_array = test_images_array/255.0


#plt.imshow(train_images_array[100], cmap=plt.cm.binary)
#plt.show()
#print(train_images_array[100])

print("shapes_of_data")
print(np.shape(train_images_array))
print(np.shape(train_labels_array))
print(np.shape(test_images_array))
print(np.shape(test_labels_array))

print("....................")



#3D to 2D, (2D to 1D)
new_train_images_array = train_images_array[:,:,0]

for i in range(1,np.shape(train_images_array)[1]):
    new_train_images_array = np.append(new_train_images_array,train_images_array[:,:,i], axis=1)


new_test_images_array = test_images_array[:,:,0]

for i in range(1,np.shape(test_images_array)[1]):
    new_test_images_array = np.append(new_test_images_array,test_images_array[:,:,i], axis=1)



print("shapes of new data")
print(np.shape(new_train_images_array))
print(np.shape(train_labels_array))
print(np.shape(new_test_images_array))
print(np.shape(test_labels_array))
print("....................")


#Split data into minibatches

batches_amount = 600;
batches = np.array_split(new_train_images_array, batches_amount, axis=0)

print("shape_of_batch")
print(np.shape(batches[5]))
print("...........................")

#defining the size of the network

"layers_size"
in_size = np.shape(new_train_images_array)[1]
h1_size = 20;
out_size = 10;



#changing the format of output data

out_train_labels_array = np.zeros((out_size,1))
out_train_labels_array[train_labels_array[0]] = 1
for i in range(1,np.shape(train_labels_array)[0]):
    temp_out_train_labels_array = np.zeros((out_size,1))
    temp_out_train_labels_array[train_labels_array[i]] = 1
    out_train_labels_array = np.append(out_train_labels_array,temp_out_train_labels_array, axis=1)

#print("out_train_labels_array")
#print(out_train_labels_array)
#print(train_labels_array)
#print(np.shape(out_train_labels_array))

out_train_labels_array = out_train_labels_array.T

#Split labels into minibatches
batches_out = np.array_split(out_train_labels_array, batches_amount, axis=0)
#print(np.shape(batches_out[5]))

out_test_labels_array = np.zeros((out_size,1))
out_test_labels_array[test_labels_array[0]] = 1
for i in range(1,np.shape(test_labels_array)[0]):
    temp_out_test_labels_array = np.zeros((out_size,1))
    temp_out_test_labels_array[test_labels_array[i]] = 1
    out_test_labels_array = np.append(out_test_labels_array,temp_out_test_labels_array, axis=1)

#print("out_train_labels_array")
#print(out_test_labels_array)
#print(test_labels_array)

out_test_labels_array = out_test_labels_array.T

#print("out_test_labels_array")
#print(out_test_labels_array)





losses = np.zeros((batches_amount,1))


#initialisation

w1 = randn(in_size,h1_size)
w2 = randn(h1_size,out_size)
rate = 0.001


#testing of an untrained network
print("Start of testing")
print(".......................")

h1_lin = new_test_images_array.dot(w1)
h1 = 1/(1 + np.exp(-h1_lin))
out_pred_lin = h1.dot(w2)
out_pred = 1/(1 + np.exp(-out_pred_lin))


print("Shapes of out")
print(np.shape(out_pred))  
print(".......................")  

out_winners = np.zeros((np.shape(out_pred)[0],1))
for i in range(np.shape(out_winners)[0]):
    out_winners[i] = np.max(out_pred[i,:])


score = 0;

for i in range(np.shape(out_winners)[0]):
    if np.where(out_pred[i,:] == out_winners[i]) == np.where(out_test_labels_array[i,:] == 1):
        score += 1
    else:
        score += 0
    
accuracy = score/np.shape(out_pred)[0];



print("accuracy")
print(accuracy)
print(".........................")



print("Start of training")
print(".......................") 
for i in range(batches_amount):
    #predinction
    h1_lin = batches[i].dot(w1)
    h1 = 1/(1 + np.exp(-h1_lin))
    out_pred_lin = h1.dot(w2)
    out_pred = 1/(1 + np.exp(-out_pred_lin))
    loss_function = np.square(out_pred - batches_out[i]).sum()

    losses[i] = loss_function

    #backpropagation

    grad_out_pred = 2.0*(out_pred - batches_out[i])

    grad_out_pred_lin = grad_out_pred*out_pred*(1-out_pred)
    grad_w2 = h1.T.dot(grad_out_pred_lin)
    grad_h1 = grad_out_pred_lin.dot(w2.T)

    grad_h1_lin = grad_h1*h1*(1-h1)
    grad_w1 = batches[i].T.dot(grad_h1_lin)

    w1 -= rate*grad_w1
    w2 -= rate*grad_w2



#testing of trained network

print("Start of testing")
print(".......................") 
h1_lin = new_test_images_array.dot(w1)
h1 = 1/(1 + np.exp(-h1_lin))
out_pred_lin = h1.dot(w2)
out_pred = 1/(1 + np.exp(-out_pred_lin))


print("Shapes of out")
print(np.shape(out_pred))   
print(".......................") 

out_winners = np.zeros((np.shape(out_pred)[0],1))
for i in range(np.shape(out_winners)[0]):
    out_winners[i] = np.max(out_pred[i,:])


score = 0;

for i in range(np.shape(out_winners)[0]):
    if np.where(out_pred[i,:] == out_winners[i]) == np.where(out_test_labels_array[i,:] == 1):
        score += 1
    else:
        score += 0
    
accuracy = score/np.shape(out_pred)[0];







print("accuracy")
print(accuracy)
print(".........................")
  

#Plot a loss descent 
plt.plot(np.arange(batches_amount),losses)
plt.show()  






