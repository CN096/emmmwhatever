import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np

# 加载Fashion-MNIST数据集
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

# 只选择指定的五类数据
selected_classes = [0, 1, 2, 3, 4]
train_images = train_images[np.isin(train_labels, selected_classes)]
train_labels = train_labels[np.isin(train_labels, selected_classes)]
test_images = test_images[np.isin(test_labels, selected_classes)]
test_labels = test_labels[np.isin(test_labels, selected_classes)]

# 对标签进行独热编码
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=len(selected_classes))
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=len(selected_classes))

# 归一化像素值到0-1之间
train_images = train_images / 255.0
test_images = test_images / 255.0

# 创建模型
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(selected_classes), activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 调整输入数据的形状
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# 训练模型
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# 保存模型
model.save('fashion_mnist_cnn_model.h5')
 
# 打印模型摘要
model.summary()
