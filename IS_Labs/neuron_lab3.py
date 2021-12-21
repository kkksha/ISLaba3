import tensorflow as tf
from tensorflow import keras
import numpy as np
import data_creation as dc 

model = keras.Sequential([
    # выравнивание входного массива в вектор
    keras.layers.Flatten(input_shape=(2,)),
    # 2 и 3 уровень сети состоят из 20 узлов
    # функция активации relu - выпрямленная линейная единица
    keras.layers.Dense(20, activation=tf.nn.relu),
	  keras.layers.Dense(20, activation=tf.nn.relu),
    # т.к. ожидается 1 выходное значение (прогнозируемое значение, т.к. это регресионная модель), следовательно только один выходной узел
    keras.layers.Dense(1)
])

# компиляция сети
# adam - функция оптимизации (оптимизатор на основе импульса и предотвращает застревание модели в локальных минимумах)
# mse - функция потерь (среднеквадратическая ошибка) Квадратичная разница между прогнозируемым и фактическим значением
# mae - средняя абсолютная ошибка
model.compile(optimizer='adam', 
              loss='mse',
              metrics=['mae'])

# обучение сетей
# обучающий набор будет подаваться в сеть 10 раз (мало эпох- недоученность, много эпох - переобучение)
model.fit(dc.train_data, dc.train_targets, epochs=10, batch_size=1)

# оценивание обученной модели на тестовом наборе данных
test_loss, test_acc = model.evaluate(dc.test_data, dc.test_targets)
# вывод значения точности теста
print('Test accuracy:', test_acc)

# подставление реальных значений
# два набора значений
a = np.array([[1000, 3000],[1,5]])
print(model.predict(a))