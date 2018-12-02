from keras.applications.inception_resnet_v2 import InceptionResNetV2

model = InceptionResNetV2(weights="imagenet",include_top=False, input_shape=(224,224,3))
model.summary()
