import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image
#print(tf.test.is_gpu_available())

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.figure(figsize=(10,10))
def show_samples(x_train,y_train,num=25):
    for i in range(num):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_train[i],cmap='gray')
        plt.xlabel(y_train[i])
    plt.show()
def load_mnist_data():
    file_path = os.path.join('data_set_tf2', 'mnist.npz')
    data=np.load(file_path)
    x_train=data['x_train']
    y_train=data['y_train']
    x_test=data['x_test']
    y_test=data['y_test']
    data.close()
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return (x_train, y_train), (x_test, y_test)
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model
def train(model, x_train, y_train):
    model.fit(x_train, y_train, epochs=4, batch_size=40)
def eva(model, x_test, y_test):
    test_loss = model.evaluate(x_test, y_test)
    print(f'Test Loss: {test_loss}')
    return test_loss
def read_image(img_path):
    img = Image.open(img_path).convert('L')  # 转为灰度图
    img = img.resize((28, 28))  # 调整大小为28x28
    img = np.array(img)  # 转为numpy数组
    img =1- img.astype('float32') / 255.0  # 归一化
    img = np.expand_dims(img, axis=0)  # 增加批次维度
    return img
def predict_and_show(model, img):
    predictions = model.predict(img)
    show_samples(img,predictions.argmax(axis=1),num=1)
def launch():
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    model = build_model()
    y_train=tf.keras.utils.to_categorical(y_train,num_classes=10)
    y_test=tf.keras.utils.to_categorical(y_test,num_classes=10)
    train(model, x_train[:60000], y_train[:60000])
    if eva(model, x_test[:10000], y_test[:10000])<0.02:
        model.save('mnist_model.h5')
    else:
        print('准确率不够')
if __name__ == '__main__':
    if os.path.exists('mnist_model.h5')==False:
        launch()
    model = tf.keras.models.load_model('mnist_model.h5')
    img=read_image(os.path.join('7.png'))
    predict_and_show(model, img)
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    predictions = model.predict(x_test[:10000])
    show_samples(x_test[:25], np.argmax(predictions[:25], axis=1))