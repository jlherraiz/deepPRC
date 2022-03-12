# -*- coding: utf-8 -*-

def scanner_INVEON_Deep_PRC(name, model): 

  # 0 - IMPORT LIBRARIES
  import tensorflow as tf
  tf.version.VERSION
  import numpy as np
  import os
  import time
  import matplotlib.pyplot as plt
  from IPython.display import clear_output
  import cv2
  import sklearn.model_selection as sk
  import tensorflow_addons as tfa

  # PARAMETROS DE LAS IMAGENES
  Nx0 = 128
  Ny0 = 128
  Nz0 = 159

  Nt_MOBY = 7
  Nt_DIGI = 7
  Nt = Nt_MOBY + Nt_DIGI
  dx = 0.776
  dy = 0.776
  dz = 0.796    #VOXEL SIZE (cm) ALONG X,Y,Z 

  # EXTENDED FOV TO MAKE IT MORE DIVISIBLE (PADDED WITH ZEROS)
  Nx1 = Nx0 
  Ny1 = Ny0 
  Nz1 = 161

  ksize_planes = 9
  ksize_rows = 32
  ksize_cols = 32

  stride_planes = 4 
  strides_rows = 16
  strides_cols = 16

  ksizes = [1, ksize_planes, ksize_rows, ksize_cols, 1]         #The size of the sliding window for each dimension of input.
  strides = [1, stride_planes, strides_rows, strides_cols, 1]   #How far the centers of two consecutive patches are in input. 
  padding = "VALID"

  # EXTENDED FOV 
  # FROM EACH PATCH WE EXTRACT 16X16X1 OUTPUT --> THEFORE WE NEED ADDITIONAL 8X8X4 IN EACH EDGE
  outputx = ksize_rows//2  #16
  outputy = ksize_cols//2  #16
  outputz = 1
  Zc = ksize_planes//2 

  extrax = ksize_rows - outputx
  extray = ksize_cols - outputy
  extraz = ksize_planes - 1

  Nx2=Nx1 + extrax
  Ny2=Ny1 + extray
  Nz2=Nz1 + extraz

  Npatchesx = Nx1 // outputx
  Npatchesy = Ny1 // outputy


  # LOAD INPUT IMAGE 
  path_img =  name # '/content/' +
  file = np.fromfile(path_img, dtype='float32')
  img = file.reshape((Nz0,Ny0,Nx0)) # Axial view 

  # RESTORING DE MODEL 

  model = tf.keras.models.load_model(model,compile=False)   # LOADING KERAS MODEL
  opt = tfa.optimizers.RectifiedAdam(lr=1e-3)
  opt = tfa.optimizers.Lookahead(opt)
  model.compile(optimizer=opt,loss='MeanAbsoluteError') 

  # ADAPTING INPUT --> PATCHES 
  import time
  t0 = time.time()

  TGa68 = img
  TGa68_tf = tf.convert_to_tensor(TGa68, tf.float32)
  TGa68_tf = tf.expand_dims(TGa68_tf,axis=0) 
  Tstride_planes = 1
  TGa68_tf = tf.image.pad_to_bounding_box(TGa68_tf,stride_planes,0,Nz2,Nx0)
  TGa68_tf = tf.transpose(TGa68_tf, perm=[0,2,3,1])
  TGa68_tf = tf.image.pad_to_bounding_box(TGa68_tf,strides_rows//2,strides_cols//2,Nx2,Ny2)
  TGa68_tf = tf.transpose(TGa68_tf, perm=[0,3,1,2])
  TGa68_tf = tf.expand_dims(TGa68_tf,axis=-1)                     # 5-D Tensor with shape [batch, in_planes, in_rows, in_cols, depth].
  #print("TGa68 initial shape= ",TGa68.shape)
  Tstrides = [1, Tstride_planes, strides_rows, strides_cols, 1]   # How far the centers of two consecutive patches are in input. 
  TGa68_tf_patches = tf.extract_volume_patches(TGa68_tf,ksizes,Tstrides,padding)   #Extract patches from input and put them in the "depth" output dimension.
  #print("TGa68_patch_size= ",TGa68_tf_patches.shape)

  TGa = tf.reshape(TGa68_tf_patches,[-1,TGa68_tf_patches.shape[4]])
  TGa_max = tf.math.reduce_max(TGa,1)+tf.constant(1.0e-5)
  TGa_max = tf.expand_dims(TGa_max,axis=-1)
  Tmaximo = tf.tile(TGa_max,tf.constant([1,TGa.shape[1]], tf.int32))

  TGa = tf.divide(TGa,Tmaximo)
  TGa = tf.reshape(TGa,[TGa.shape[0],ksize_planes,ksize_rows,ksize_cols])
  TGa = tf.transpose(TGa, perm=[0, 2, 3, 1])
  TGa.shape

  inp_tf = TGa  


  # NORMALIZATION INPUT 
  Tmaxim2 = tf.tile(TGa_max,tf.constant([1,ksize_rows*ksize_cols//4], tf.int32))
  Tmaxim2 = tf.reshape(Tmaxim2,[Tmaxim2.shape[0],ksize_rows//2,ksize_cols//2,1])
  Tmaxim2.shape

  # MODEL PREDICT 
  print('Start Deep-PRC')
  t1 = time.time()
  TEst = model.predict(inp_tf)
  t2 = time.time()
  TEst = tf.image.central_crop(TEst,0.5)   # KEEP JUST THE CENTRAL PART

  TEst = tf.multiply(TEst,Tmaxim2)  #Restore Normalization    #SHAPE 8100,16,16,1
  TEst2 = tf.reshape(TEst,[-1,Npatchesy,Npatchesx,16,16])     #SHAPE 81X10X10X16X16
  TEst3 = tf.transpose(TEst2,perm=[0,1,3,2,4])                #SHAPE 81X10X16X10X16   (REMEMBER: VOLUME [Z,Y,X] ORDER)
  TEst3 = tf.reshape(TEst3,[-1,Ny1,Nx1])  

  d = np.squeeze(np.array(TEst3[1:Nz0+1,:,:],'float32'))
  f=open("Corrected.raw","wb")
  f.write(d)
  f.close()
  
  print('Done!')

  return print('Look at your /content folder to download your corrected image')